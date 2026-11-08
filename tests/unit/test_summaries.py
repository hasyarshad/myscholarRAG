"""Document-level summaries and review detection.

A survey's methods section reads like any other, so chunk retrieval alone cannot tell
a method the author reviewed from one the author proposed.
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
        assert not looks_like_review(title)


class TestRendering:
    def test_the_summary_names_techniques_twice(self) -> None:
        """Once as the paper names them and once in the general terms a reader would use."""
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
        text = render(
            PaperSummary(topic="Surveys online learning algorithms."),
            title="An Overview of Sequential Learning",
            is_review=True,
        )

        assert "other researchers" in text

    def test_the_title_leads(self) -> None:
        text = render(PaperSummary(topic="Something."), title="A Specific Title")

        assert text.startswith("A Specific Title")


class TestBlockRendering:
    def test_a_review_block_is_marked_in_the_prompt(self) -> None:
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
