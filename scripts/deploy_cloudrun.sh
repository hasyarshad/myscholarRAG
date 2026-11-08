#!/usr/bin/env bash
# Build and deploy the API to Google Cloud Run.
#
# Cloud Run scales to zero, so an idle assistant costs nothing. The price is a
# cold start while the container boots and loads the 416MB embedder; the proxy in
# web/app/api/scholar allows 90 seconds for it.
#
#   ./scripts/deploy_cloudrun.sh --dry-run
#   ./scripts/deploy_cloudrun.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

# Always succeeds: under `set -e` a grep with no match (no .env yet) would abort silently.
value() { grep -E "^$1=" .env 2>/dev/null | cut -d= -f2- | tr -d '"'"'"'' || true; }

PROJECT="${GCP_PROJECT_ID:-$(value GCP_PROJECT_ID)}"
REGION="${GCP_REGION:-$(value GCP_REGION)}"
REGION="${REGION:-northamerica-northeast1}"
SERVICE="${GCP_SERVICE:-$(value GCP_SERVICE)}"
SERVICE="${SERVICE:-myscholar-api}"

# Runtime identity with read access to the three secrets below and nothing else.
# The default Compute Engine account needs the Compute API and is shared project-wide.
RUNTIME_SA="${GCP_RUNTIME_SA:-$(value GCP_RUNTIME_SA)}"
RUNTIME_SA="${RUNTIME_SA:-myscholar-run@$PROJECT.iam.gserviceaccount.com}"

# Build identity. Projects created from 2025 onward have no legacy Cloud Build
# account, and `gcloud builds submit` without `--service-account` then fails with
# PERMISSION_DENIED, even for a project owner.
BUILD_SA="${GCP_BUILD_SA:-$(value GCP_BUILD_SA)}"
BUILD_SA="${BUILD_SA:-myscholar-build@$PROJECT.iam.gserviceaccount.com}"
if [[ -z "$PROJECT" ]]; then
    echo "Set GCP_PROJECT_ID in .env (see .env.example)."
    exit 1
fi

IMAGE="$REGION-docker.pkg.dev/$PROJECT/myscholar/api"

cat <<INFO
project : $PROJECT
region  : $REGION
service : $SERVICE
runtime : $RUNTIME_SA
build   : $BUILD_SA
image   : $IMAGE
model   : exported and verified during the build, not committed
INFO

# Every question makes several database round trips, so warn (without failing)
# when the database and the service look like different continents.
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

# The SDK adds itself to the login shell's profile, which a non-login shell skips.
if ! command -v gcloud >/dev/null; then
    for candidate in "$HOME/google-cloud-sdk/bin" "/usr/lib/google-cloud-sdk/bin" "/snap/bin"; do
        [[ -x "$candidate/gcloud" ]] && { export PATH="$candidate:$PATH"; break; }
    done
fi
command -v gcloud >/dev/null || { echo "gcloud not installed: https://cloud.google.com/sdk"; exit 1; }

gcloud config set project "$PROJECT" >/dev/null

# One-time setup. Everything below is idempotent, so the script is safe to re-run.
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
    cloudbuild.googleapis.com secretmanager.googleapis.com

# Grants are checked here because a deploy that cannot read its secrets fails
# only after the eight-minute build.
gcloud iam service-accounts describe "$RUNTIME_SA" >/dev/null 2>&1 || \
    gcloud iam service-accounts create "${RUNTIME_SA%%@*}" \
        --display-name="myScholar Cloud Run runtime"

if ! gcloud iam service-accounts describe "$BUILD_SA" >/dev/null 2>&1; then
    gcloud iam service-accounts create "${BUILD_SA%%@*}" \
        --display-name="myScholar Cloud Build"
    for role in roles/cloudbuild.builds.builder roles/logging.logWriter \
                roles/artifactregistry.writer; do
        gcloud projects add-iam-policy-binding "$PROJECT" \
            --member="serviceAccount:$BUILD_SA" --role="$role" --quiet >/dev/null
    done
fi
for secret in myscholar-db-url myscholar-llm-key myscholar-shared-secret; do
    if ! gcloud secrets describe "$secret" >/dev/null 2>&1; then
        echo "missing secret: $secret. Create it before deploying:"
        echo "  printf '%s' '<value>' | gcloud secrets create $secret --data-file=-"
        exit 1
    fi
    gcloud secrets add-iam-policy-binding "$secret" \
        --member="serviceAccount:$RUNTIME_SA" \
        --role=roles/secretmanager.secretAccessor --quiet >/dev/null
done
gcloud artifacts repositories describe myscholar --location="$REGION" >/dev/null 2>&1 || \
    gcloud artifacts repositories create myscholar --repository-format=docker --location="$REGION"

# Built by Cloud Build, so no local Docker is needed. The build exports the
# embedder, hence the larger machine and the 20-minute timeout (it usually takes eight).
# A real file because `--config=-` fails with "Unable to read file [-]" on some
# gcloud versions.
BUILD_CONFIG="$(mktemp -t myscholar-cloudbuild-XXXXXX.yaml)"
trap 'rm -f "$BUILD_CONFIG"' EXIT
cat > "$BUILD_CONFIG" <<YAML
steps:
  - name: gcr.io/cloud-builders/docker
    args: ['build', '-f', 'docker/Dockerfile', '-t', '$IMAGE', '.']
images: ['$IMAGE']
timeout: 1200s
options:
  machineType: E2_HIGHCPU_8
YAML

# Required with a user-specified build service account, which needs a
# project-owned logs bucket.
gcloud builds submit \
    --config="$BUILD_CONFIG" \
    --service-account="projects/$PROJECT/serviceAccounts/$BUILD_SA" \
    --default-buckets-behavior=regional-user-owned-bucket \
    .

# --cpu-boost shortens the cold start that --min-instances=0 brings. Concurrency is
# 8 (default 80) because each request holds an ONNX session and waits on a hosted
# LLM; the app's Gate limits it further.
gcloud run deploy "$SERVICE" \
    --image="$IMAGE" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --service-account="$RUNTIME_SA" \
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
echo "checking readiness (a cold start takes a moment)..."
# /readyz checks the database and a non-empty index. /healthz cannot be used:
# Google's frontend intercepts it and returns its own 404.
curl -fsS --retry 5 --retry-delay 5 --retry-all-errors "$URL/readyz" && echo
echo
echo "Now set these in Vercel, on the site project:"
echo "  MYSCHOLAR_API_URL=$URL"
echo "  MYSCHOLAR_API_SECRET=<the same value as the myscholar-shared-secret secret>"
