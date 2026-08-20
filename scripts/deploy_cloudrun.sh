#!/usr/bin/env bash
# Build and deploy the API to Google Cloud Run.
#
# Cloud Run is the fit here for one reason: it scales to zero. A portfolio
# assistant is idle almost always, and paying for an always-on container to
# answer forty questions a month is the wrong shape. The free tier is 2M
# requests, 360k GB-seconds and 180k vCPU-seconds monthly, which this cannot
# plausibly exhaust.
#
# The cost of scaling to zero is a cold start: the first request after an idle
# period waits for the container to start and the 416MB embedder to load. The
# Next.js proxy in web/app/api/scholar handles that with a real message rather
# than a failure, which is why it has a 90-second timeout.
#
#   ./scripts/deploy_cloudrun.sh --dry-run
#   ./scripts/deploy_cloudrun.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

# Always succeeds. Under `set -e` a grep that matches nothing is a failed
# command, and inside an assignment that aborts the script with no message —
# which is exactly what "no .env yet" looks like on a first run.
value() { grep -E "^$1=" .env 2>/dev/null | cut -d= -f2- | tr -d '"'"'"'' || true; }

PROJECT="${GCP_PROJECT_ID:-$(value GCP_PROJECT_ID)}"
REGION="${GCP_REGION:-$(value GCP_REGION)}"
REGION="${REGION:-europe-west2}"
SERVICE="${GCP_SERVICE:-$(value GCP_SERVICE)}"
SERVICE="${SERVICE:-myscholar-api}"
if [[ -z "$PROJECT" ]]; then
    echo "Set GCP_PROJECT_ID in .env — see .env.example."
    exit 1
fi

IMAGE="$REGION-docker.pkg.dev/$PROJECT/myscholar/api"

cat <<INFO
project : $PROJECT
region  : $REGION
service : $SERVICE
image   : $IMAGE
model   : exported and verified during the build, not committed
INFO

# Every question makes several database round trips, so putting the container
# and the database on different continents is felt directly. Reported rather
# than enforced: there are good reasons to accept it, but none to be surprised.
DB_HOST="$(value SUPABASE_DB_URL | sed -E 's#.*@##; s#[:/].*##')"
if [[ -n "$DB_HOST" ]]; then
    echo "database: $DB_HOST"
    case "$DB_HOST:$REGION" in
        *ca-central*:northamerica-*|*us-east*:us-*|*us-west*:us-*|\
        *eu-west*:europe-*|*eu-central*:europe-*|*ap-south*:asia-*)
            echo "          (same continent as $REGION)" ;;
        *)
            echo
            echo "WARNING: the database and Cloud Run look like different regions."
            echo "         Each question makes several queries, so a cross-continent"
            echo "         hop adds roughly 0.5-1s to every answer."
            echo "         Closest to $DB_HOST would usually be:"
            case "$DB_HOST" in
                *ca-central*) echo "           GCP_REGION=northamerica-northeast1  (Montreal)" ;;
                *us-east*)    echo "           GCP_REGION=us-east1" ;;
                *us-west*)    echo "           GCP_REGION=us-west1" ;;
                *eu-west*)    echo "           GCP_REGION=europe-west2  (London)" ;;
                *)            echo "           a region on the same continent" ;;
            esac
            echo ;;
    esac
fi

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "dry run: nothing built or deployed"
    exit 0
fi

# The SDK installs to ~/google-cloud-sdk and adds itself to the login shell's
# profile, which a non-login shell never reads. Look there before giving up.
if ! command -v gcloud >/dev/null; then
    for candidate in "$HOME/google-cloud-sdk/bin" "/usr/lib/google-cloud-sdk/bin" "/snap/bin"; do
        [[ -x "$candidate/gcloud" ]] && { export PATH="$candidate:$PATH"; break; }
    done
fi
command -v gcloud >/dev/null || { echo "gcloud not installed: https://cloud.google.com/sdk"; exit 1; }

gcloud config set project "$PROJECT" >/dev/null

# One-time setup, all idempotent, so this script is safe to re-run.
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories describe myscholar --location="$REGION" >/dev/null 2>&1 || \
    gcloud artifacts repositories create myscholar --repository-format=docker --location="$REGION"

# Built by Cloud Build rather than locally: there is no Docker on this machine
# (no root, and rootless needs setuid helpers that are not installed).
# The build exports the embedder, so it needs more time and more machine than a
# plain pip install would. Twenty minutes is generous; it usually takes eight.
gcloud builds submit --config=- <<YAML
steps:
  - name: gcr.io/cloud-builders/docker
    args: ['build', '-f', 'docker/Dockerfile', '-t', '$IMAGE', '.']
images: ['$IMAGE']
timeout: 1200s
options:
  machineType: E2_HIGHCPU_8
YAML

# --min-instances=0 is the free tier. --cpu-boost shortens the cold start that
# choice buys, by giving the container extra CPU while it loads the model.
#
# Concurrency 8 rather than the default 80: each request holds an ONNX session
# and waits on a hosted LLM, and 80 of those on two vCPU is a queue pretending
# to be parallelism. The app's own Gate limits it further.
gcloud run deploy "$SERVICE" \
    --image="$IMAGE" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=3 \
    --concurrency=8 \
    --timeout=120 \
    --cpu-boost \
    --set-env-vars=MYSCHOLAR_DATABASE__MIGRATE_ON_STARTUP=false \
    --set-secrets=MYSCHOLAR_DATABASE__URL=myscholar-db-url:latest,MYSCHOLAR_LLM__API_KEY=myscholar-llm-key:latest,MYSCHOLAR_API__SHARED_SECRET=myscholar-shared-secret:latest

URL=$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')
echo
echo "deployed: $URL"
echo "checking health (a cold start takes a moment)..."
curl -fsS --retry 5 --retry-delay 5 --retry-all-errors "$URL/healthz" && echo
echo
echo "Now set these in Vercel, on the site project:"
echo "  MYSCHOLAR_API_URL=$URL"
echo "  MYSCHOLAR_API_SECRET=<the same value as the myscholar-shared-secret secret>"
