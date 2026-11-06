"""SaveMV / ReCo: late-interaction (ColBERT) retrieval without storing token vectors.

Each document keeps only a centroid-bag sketch; queries are ranked by a MaxSim estimate from the sketch,
the best candidates are re-encoded from raw text (recomputed), and a calibrated rule decides when to stop.

    core/         shared foundation: token store, exact MaxSim, codebooks, metrics
    reco/         the method (sketch, estimate, stopping, online engine)
    baselines/    compared methods
    encoders/     ColBERT encoder
    data/         BEIR and agent-memory benchmarks
    experiments/  experiment drivers (python -m savemv run | bench | memory)
"""

__version__ = "0.2.0"
