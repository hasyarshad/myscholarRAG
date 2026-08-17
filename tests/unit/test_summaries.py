"""Document-level description, and the review distinction.

Chunk retrieval cannot tell "he wrote a survey about X" from "he invented X",
because a survey's methods section reads exactly like a methods section. The
assistant credited Haseeb with proposing GGAP-RBF on that basis — an algorithm
his review merely described — which is the worst error a portfolio assistant
can make and the most authoritative-sounding.
"""

from __future__ import annotations

import pytest

from myscholar.ingest.summaries import PaperSummary, looks_like_review, render


class TestReviewDetection:
    @pytest.mark.parametrize(
        "title",
        [
            "An Overview of Sequential Learning Algorithms for Single Hidden Layer Networks",
            "Human Activity Recognition: Review, Taxonomy and Open Challenges",
            "A Survey of Model Predictive Control",
            "Recent Advances in Fault Diagnosis",
            "Current Issues and Future Trends in Online Learning",
            "A Comparative Study of Sliding Mode Controllers",
        ],
    )
    def test_a_survey_is_recognised_from_its_title(self, title: str) -> None:
        """The model was asked first and would not say.

        Given a paper called "An Overview of ... Current Issues & Future
        Trends", a 4B summariser still reported it as original work. The title
        is the more reliable signal and cannot drift.
        """
        assert looks_like_review(title)

    @pytest.mark.parametrize(
        "title",
        [
            "Rotor imbalance fault classification through unified feature fusion",
            "Leveraging Meta-Learner-Based Stacking Framework for Stator Winding Fault Detection",
            "Packet-Level Network Intrusion Detection Based on Reinforcement Learning",
            "Weighting Factors Optimization of Model Predictive Torque Control",
        ],
    )
    def test_original_work_is_not(self, title: str) -> None:
        """A false positive costs a credit he has earned, so the pattern names
        the words that announce a survey rather than guessing."""
        assert not looks_like_review(title)


class TestRendering:
    def test_the_summary_names_techniques_twice(self) -> None:
        """Once as the paper names them, once as a reader would.

        This is the whole point: someone asking about machine learning will not
        search for "AFL-1DCNN", and the papers that use it never say "machine
        learning".
        """
        text = render(
            PaperSummary(
                topic="Detecting stator winding faults from sound.",
                methods=["AFL-1DCNN (convolutional neural network)"],
                fields=["machine learning", "fault diagnosis"],
                findings="Reached 98.5% accuracy.",
            ),
            title="Leveraging a Stacking Framework",
        )

        assert "AFL-1DCNN" in text
        assert "convolutional neural network" in text
        assert "machine learning" in text

    def test_a_review_says_so_in_its_summary(self) -> None:
        """The summary is embedded, so this also keeps a survey from matching
        "his contribution to X" as strongly as the papers that made one."""
        text = render(
            PaperSummary(topic="Surveys online learning algorithms."),
            title="An Overview of Sequential Learning",
            is_review=True,
        )

        assert "other researchers" in text

    def test_the_title_leads(self) -> None:
        """The most reliable sentence about a paper, and what makes the summary
        a description of *this* document rather than free-floating prose."""
        text = render(PaperSummary(topic="Something."), title="A Specific Title")

        assert text.startswith("A Specific Title")


class TestBlockRendering:
    def test_a_review_block_is_marked_in_the_prompt(self) -> None:
        """A passage cannot show that its paper is a survey, so the marker is
        the only thing standing between the reader and a misattribution."""
        from uuid import uuid4

        from myscholar.domain.documents import ContextBlock

        block = ContextBlock(
            label="C1",
            document_id=uuid4(),
            section_id=uuid4(),
            document_title="An Overview of Sequential Learning",
            section_heading="III. Extreme Learning Machines",
            text="Huang proposed the OS-ELM algorithm.",
            page_start=3,
            page_end=3,
            is_review=True,
        )

        assert "REVIEW" in block.render()

    def test_an_ordinary_block_is_not_marked(self) -> None:
        from uuid import uuid4

        from myscholar.domain.documents import ContextBlock

        block = ContextBlock(
            label="C1",
            document_id=uuid4(),
            section_id=uuid4(),
            document_title="A Stacking Framework",
            section_heading="Methods",
            text="We propose a GBM meta-learner.",
            page_start=3,
            page_end=3,
        )

        assert "REVIEW" not in block.render()
