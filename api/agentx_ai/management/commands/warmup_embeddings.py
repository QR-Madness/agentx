"""
Management command to warm up embedding models.

Downloads and initializes the embedding model so first requests don't have cold start latency.
Useful for production deployments where you want models ready before handling traffic.

Usage:
    python manage.py warmup_embeddings              # Warm up embedding model
    python manage.py warmup_embeddings --validate   # Also validate dimensions match config
"""

import time
from django.core.management.base import BaseCommand

from agentx_ai.kit.agent_memory.embeddings import get_embedder
from agentx_ai.kit.agent_memory.config import get_settings


class Command(BaseCommand):
    help = "Warm up embedding models for production readiness"

    def add_arguments(self, parser):
        parser.add_argument(
            "--validate",
            action="store_true",
            help="Validate embedding dimensions match configuration",
        )

    def handle(self, *args, **options):
        settings = get_settings()
        self.stdout.write(f"Embedding provider: {settings.embedding_provider}")

        if settings.embedding_provider == "openai":
            self.stdout.write(f"Model: {settings.embedding_model}")
            self.stdout.write(
                self.style.WARNING(
                    "OpenAI embeddings don't require local warm-up (API-based)."
                )
            )
            if options["validate"]:
                self.stdout.write(
                    f"Configured dimensions: {settings.embedding_dimensions}"
                )
            return

        # Local embedding model warm-up
        model_name = settings.local_embedding_model
        self.stdout.write(f"Model: {model_name}")
        self.stdout.write("Loading embedding model...")

        start_time = time.time()
        embedder = get_embedder()

        # Force model initialization by running a test embedding
        test_text = "Warming up embedding model for production deployment."
        embedding = embedder.embed_single(test_text)

        load_time = time.time() - start_time

        self.stdout.write(
            self.style.SUCCESS(f"✓ Model loaded in {load_time:.2f}s")
        )
        self.stdout.write(f"  Output dimensions: {len(embedding)}")

        if options["validate"]:
            actual, configured, match = embedder.validate_dimensions()
            if match:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"✓ Dimensions match: {actual} (actual) == {configured} (configured)"
                    )
                )
            else:
                self.stdout.write(
                    self.style.ERROR(
                        f"✗ Dimension mismatch: {actual} (actual) != {configured} (configured)"
                    )
                )
                self.stdout.write(
                    self.style.WARNING(
                        f"  Update EMBEDDING_DIMENSIONS={actual} or re-initialize schemas"
                    )
                )
