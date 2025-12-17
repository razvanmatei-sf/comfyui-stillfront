/**
 * Dynamic Image Inputs for SF VertexAI Nano Banana Pro Edit
 * Uses image_count dropdown and "Update inputs" button to show/hide image slots.
 * Supports up to 14 images as per Gemini 3 Pro Image API.
 */

import { app } from "../../../scripts/app.js";

const MAX_IMAGES = 14;
const DEFAULT_IMAGES = 3;
const NODE_NAME = "SFVertexAINanaBananaProEdit";

app.registerExtension({
  name: "Stillfront.NanaBananaDynamicInputs",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== NODE_NAME) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated
        ? onNodeCreated.apply(this, arguments)
        : undefined;

      // Add "Update inputs" button
      this.addWidget("button", "Update inputs", null, () => {
        this._updateImageInputs();
      });

      // Initial update to show only default number of inputs
      setTimeout(() => {
        this._updateImageInputs();
      }, 100);

      return r;
    };

    nodeType.prototype._updateImageInputs = function () {
      if (!this.inputs) return;

      // Get the target number of inputs from the image_count widget
      const imageCountWidget = this.widgets.find(
        (w) => w.name === "image_count",
      );
      const targetCount = imageCountWidget
        ? imageCountWidget.value
        : DEFAULT_IMAGES;

      // Get current image inputs
      const imageInputs = this.inputs.filter(
        (input) => input.name && input.name.startsWith("image_"),
      );

      const currentCount = imageInputs.length;

      // If we need fewer inputs, remove from the end (only if not connected)
      if (targetCount < currentCount) {
        for (let i = this.inputs.length - 1; i >= 0; i--) {
          const input = this.inputs[i];
          if (input.name && input.name.startsWith("image_")) {
            const inputNum = parseInt(input.name.split("_")[1]);
            if (inputNum > targetCount) {
              // Only remove if not connected
              if (input.link === null) {
                this.removeInput(i);
              }
            }
          }
        }
      }
      // If we need more inputs, add them
      else if (targetCount > currentCount) {
        for (let i = currentCount + 1; i <= targetCount; i++) {
          this.addInput(`image_${i}`, "IMAGE", {
            tooltip: `Input image ${i}`,
          });
        }
      }

      // Resize the node to fit
      this.setSize(this.computeSize());
      this.setDirtyCanvas(true, true);
    };

    // Restore inputs after loading from saved workflow
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      if (onConfigure) {
        onConfigure.apply(this, arguments);
      }

      // Defer to ensure widgets are loaded
      setTimeout(() => {
        this._updateImageInputs();
      }, 100);
    };
  },
});
