"""Atomic, epoch-boundary recovery; distinct from the best PSNR model.

A committed generation contains model+Adam state, callback counters, history and
best-model snapshot. The pointer changes only after every file is complete.
Partial epochs are replayed; concurrent writers to one run are unsupported.
"""
import json
import os
from pathlib import Path
import shutil
import uuid
import tensorflow as tf


class Recovery(tf.keras.callbacks.Callback):
    FIELDS = ("best", "wait", "cooldown_counter", "stopped_epoch", "best_epoch")

    def __init__(self, output, callbacks, state=None):
        super().__init__()
        self.output = Path(output)
        self.callbacks = callbacks
        self.state = state

    def on_train_begin(self, logs=None):
        # Runs after monitored callbacks reset their counters.
        if self.state:
            for callback, values in zip(self.callbacks, self.state["callbacks"]):
                for key, value in values.items():
                    setattr(callback, key, float(value) if key == "best" and value is not None else value)

    def on_epoch_end(self, epoch, logs=None):
        root = self.output / "recovery"
        root.mkdir(exist_ok=True)
        generation = root / (f"epoch-{epoch+1}-" + uuid.uuid4().hex)
        generation.mkdir()
        self.model.save(generation / "latest.keras")
        shutil.copy2(self.output / "best.keras", generation / "best.keras")
        shutil.copy2(self.output / "training_history.csv", generation / "history.csv")
        states = []
        for callback in self.callbacks:
            values = {}
            for key in self.FIELDS:
                if hasattr(callback, key):
                    value = getattr(callback, key)
                    values[key] = str(float(value)) if key == "best" and value is not None else int(value) if value is not None else None
            states.append(values)
        (generation / "state.json").write_text(json.dumps({"epoch": epoch+1, "callbacks": states}))
        temporary = root / "current.tmp"
        temporary.write_text(generation.name)
        os.replace(temporary, root / "current")
        # Only delete obsolete committed generations after publishing the pointer.
        for older in root.glob("epoch-*"):
            if older != generation:
                shutil.rmtree(older)


def load_recovery(output):
    output = Path(output)
    root = output / "recovery"
    generation = root / (root / "current").read_text().strip()
    state = json.loads((generation / "state.json").read_text())
    model = tf.keras.models.load_model(generation / "latest.keras")
    shutil.copy2(generation / "best.keras", output / "best.keras")
    shutil.copy2(generation / "history.csv", output / "training_history.csv")
    return model, state
