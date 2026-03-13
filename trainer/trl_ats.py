from __future__ import annotations

from typing import Any

from trl import SFTTrainer


class ATSTrainer(SFTTrainer):
    def _should_emit_ats_status(self) -> bool:
        if not self.is_world_process_zero():
            return False
        logging_steps = getattr(self.args, "logging_steps", 0) or 0
        if logging_steps <= 0:
            return False
        next_step = self.state.global_step + 1
        return next_step % int(logging_steps) == 0

    @staticmethod
    def _metric_value(outputs: Any, key: str) -> float:
        value = getattr(outputs, key, None)
        if value is None:
            return 0.0
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "mean"):
            value = value.mean()
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        outputs = model(**inputs)
        metrics = {
            "loss": self._metric_value(outputs, "loss"),
            "ats_mean_temperature": self._metric_value(outputs, "ats_mean_temperature"),
            "ats_incorrect_token_fraction": self._metric_value(outputs, "ats_incorrect_token_fraction"),
            "ats_hard_loss": self._metric_value(outputs, "ats_hard_loss"),
            "ats_smooth_loss": self._metric_value(outputs, "ats_smooth_loss"),
            "ats_ok": self._metric_value(outputs, "ats_ok"),
        }
        self.log(metrics)
        if self._should_emit_ats_status():
            ok_text = "Yes" if metrics["ats_ok"] >= 0.5 else "No"
            print(
                "[ATS] "
                f"mean_temperature={metrics['ats_mean_temperature']:.4f} "
                f"incorrect_token_fraction={metrics['ats_incorrect_token_fraction']:.4f} "
                f"hard_loss={metrics['ats_hard_loss']:.4f} "
                f"smooth_loss={metrics['ats_smooth_loss']:.4f} "
                f"OK: {ok_text}"
            )
        if return_outputs:
            return outputs["loss"], outputs
        return outputs["loss"]
