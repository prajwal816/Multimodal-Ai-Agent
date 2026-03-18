"""
src/tools/vision_tool.py
─────────────────────────
VisionAnalysisTool — LangChain BaseTool wrapping the VisionModel.
"""

from __future__ import annotations

import logging
from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.vision.vision_model import BaseVisionModel

logger = logging.getLogger("multimodal_agent.tools.vision")


class VisionInput(BaseModel):
    image_path: str = Field(description="Absolute or relative path to the image file to analyse.")
    prompt: str = Field(
        default="Describe this image in detail, including objects, colours, and context.",
        description="Instruction for the vision model.",
    )


class VisionAnalysisTool(BaseTool):
    """
    Image analysis tool that runs the configured VisionModel
    (LLaVA or stub) and returns a detailed description.
    """

    name: str = "vision_analysis"
    description: str = (
        "Analyse an image and return a detailed description of its contents. "
        "Input must be the file path to an image (jpg, png, webp, etc.). "
        "Use this when the task involves understanding visual content."
    )
    args_schema: Type[BaseModel] = VisionInput
    vision_model: BaseVisionModel = None  # type: ignore[assignment]

    class Config:
        arbitrary_types_allowed = True

    def _run(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        if self.vision_model is None:
            return "[Error] Vision model not initialised."

        import os
        if not os.path.exists(image_path):
            # Return a graceful description for missing files
            logger.warning(f"Image not found: {image_path}. Returning stub description.")
            return (
                f"[Vision Tool] Image file '{image_path}' was not found on disk. "
                "For demonstration purposes: the image would contain visual features "
                "such as objects, colours, textures, and spatial relationships relevant to the task."
            )

        try:
            description = self.vision_model.describe(image_path=image_path, prompt=prompt)
            logger.debug(f"Vision analysis complete for {image_path!r}")
            return f"[Vision Analysis]\n{description}"
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return f"[Vision Tool Error] {e}"

    async def _arun(self, image_path: str, prompt: str = "Describe this image in detail.") -> str:
        return self._run(image_path, prompt)
