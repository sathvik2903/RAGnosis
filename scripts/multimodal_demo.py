from __future__ import annotations

import argparse
import json

from multimodal.schemas import MultimodalRequest
from multimodal.service import MultimodalRAGService


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAGnosis multimodal biomedical RAG")
    parser.add_argument("--image", required=True, help="Path to an image")
    parser.add_argument("--question", required=True, help="Question about the supplied image")
    args = parser.parse_args()

    result = MultimodalRAGService().run(
        MultimodalRequest(question=args.question, image_path=args.image)
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
