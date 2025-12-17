/**
 * Dynamic Image Inputs for SF VertexAI Nano Banana Pro Edit
 * Automatically adds new image input slots when images are connected.
 * Supports up to 14 images as per Gemini 3 Pro Image API.
 */

import { app } from "../../../scripts/app.js";

const MAX_IMAGES = 14;
const NODE_NAME = "SFVertexAINanaBananaProEdit";

app.registerExtension({
    name: "Stillfront.NanaBananaDynamicInputs",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // Track the highest image index we've added
            this._maxImageIndex = 1;

            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (side, slot, connect, link_info, output) {
            if (onConnectionsChange) {
                onConnectionsChange.apply(this, arguments);
            }

            // Only handle input connections (side === 1)
            if (side !== 1) return;

            // Check if this is an image slot
            const input = this.inputs[slot];
            if (!input || !input.name.startsWith("image_")) return;

            // Fix inputs after connection change
            this._fixImageInputs();
        };

        nodeType.prototype._fixImageInputs = function () {
            if (!this.inputs) return;

            // Count connected image inputs
            let connectedCount = 0;
            let highestConnectedIndex = 0;

            for (const input of this.inputs) {
                if (input.name && input.name.startsWith("image_")) {
                    const index = parseInt(input.name.split("_")[1]);
                    if (input.link !== null) {
                        connectedCount++;
                        highestConnectedIndex = Math.max(highestConnectedIndex, index);
                    }
                }
            }

            // Count current image inputs
            const currentImageInputs = this.inputs.filter(
                i => i.name && i.name.startsWith("image_")
            ).length;

            // We need one more empty slot than the highest connected (up to MAX_IMAGES)
            const targetInputs = Math.min(highestConnectedIndex + 1, MAX_IMAGES);

            // Add inputs if needed
            if (currentImageInputs < targetInputs) {
                for (let i = currentImageInputs + 1; i <= targetInputs; i++) {
                    this.addInput(`image_${i}`, "IMAGE", {
                        tooltip: `Input image ${i}${i < MAX_IMAGES ? " (connect to add more)" : " (max)"}`
                    });
                }
                this._maxImageIndex = targetInputs;
            }

            // Remove excess empty inputs from the end (but keep at least one)
            if (currentImageInputs > targetInputs && targetInputs >= 1) {
                // Remove inputs from highest to target, but only if they're not connected
                for (let i = this.inputs.length - 1; i >= 0; i--) {
                    const input = this.inputs[i];
                    if (input.name && input.name.startsWith("image_")) {
                        const index = parseInt(input.name.split("_")[1]);
                        if (index > targetInputs && input.link === null) {
                            this.removeInput(i);
                        }
                    }
                }
            }

            // Ensure we always have at least one image input
            const remainingImageInputs = this.inputs.filter(
                i => i.name && i.name.startsWith("image_")
            ).length;

            if (remainingImageInputs === 0) {
                this.addInput("image_1", "IMAGE", {
                    tooltip: "Input image 1 (connect to add more)"
                });
            }
        };

        // Fix inputs after loading from saved workflow
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) {
                onConfigure.apply(this, arguments);
            }

            // Defer the fix to ensure all connections are loaded
            setTimeout(() => {
                this._fixImageInputs();
            }, 100);
        };
    }
});
