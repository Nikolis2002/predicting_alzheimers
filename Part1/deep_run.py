import subprocess
import itertools
import time
import tensorflow as tf

class NeuronComboRunner:
    def __init__(self):
        self.gpu_count = len(tf.config.list_physical_devices('GPU'))
        print(f"Detected {self.gpu_count} available GPUs")

    def _clear_gpu_memory(self):
        tf.keras.backend.clear_session()
        if tf.config.list_physical_devices('GPU'):
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)

    def generate_layer_combos(self):
        neuron_choices = [32, 64, 128, 256]
        
        # Generate all valid 2-layer combinations
        for combo in itertools.product(neuron_choices, repeat=2):
            yield list(combo)
        
        # Generate all valid 3-layer combinations
        for combo in itertools.product(neuron_choices, repeat=3):
            yield list(combo)

    def generate_commands(self):
        for layers in self.generate_layer_combos():
            hidden_str = ",".join(str(n) for n in layers)
            
            cmd = [
                "python3", "alzheimers_prediction.py",
                "--optimizer", "SGD",
                "--momentum", "0.6",
                "--lr", "0.001",
                "--epochs", "1100",
                "--use_l1","True",
                "--r", "0.001",
                "--more_layers", "True",
                "--hidden_layers", hidden_str
            ]
            print(cmd)
            yield cmd

    def run_experiment(self, cmd):
        self._clear_gpu_memory()
        print(f" Running: {' '.join(cmd)}")
        start_time = time.time()

        try:
            subprocess.run(
                cmd,
                check=True,
                text=True,
                timeout=14400
            )
            print(f"Completed in {time.time()-start_time:.1f}s")
            return True

        except subprocess.TimeoutExpired:
            print(f"Timeout after {time.time()-start_time:.1f}s")
            return False

        except subprocess.CalledProcessError as e:
            print(f"Failed with code {e.returncode}")
            return False

    def run_all(self):
        total = 0
        success = 0

        for cmd in self.generate_commands():
            total += 1
            if self.run_experiment(cmd):
                success += 1
            time.sleep(1)

        print(f"\nCompleted {success}/{total} neuron architecture experiments")

if __name__ == "__main__":
    runner = NeuronComboRunner()
    runner.run_all()
