
import subprocess
import os
import time
import tensorflow as tf

class GPUExperimentRunner:
    def __init__(self):
        # GPU setup
        self.gpu_count = len(tf.config.list_physical_devices('GPU'))
        print(f"Detected {self.gpu_count} available GPUs")
        
    def _clear_gpu_memory(self):
        """Clear GPU memory between experiments"""
        tf.keras.backend.clear_session()
        if tf.config.list_physical_devices('GPU'):
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)

    def generate_commands(self):
        """Generate all command combinations"""
        # Phase 1: Test different layer configurations
        for layers in ["same", "double", "two thirds", "half"]:
            yield [
                "python3", "alzheimers_predV2.py",
                "--optimizer", "SGD",
                "--epochs", "600",
                "--num_of_layers", layers
            ]
        
        # Phase 2: Test learning rate and momentum combinations
        for lr, momentum in [(0.001,0.2), (0.001,0.6), (0.05,0.6), (0.1,0.6)]:
            yield [
                "python3", "alzheimers_predV2.py",
                "--optimizer", "SGD",
                "--epochs", "600",
                "--num_of_layers", "double",
                "--momentum", str(momentum),
                "--lr", str(lr)
            ]
        
        # Phase 3: Test r values
        for r in [0.0001, 0.001, 0.01]:
            yield [
                "python3", "alzheimers_predV2.py",
                "--optimizer", "SGD",
                "--epochs", "600",
                "--num_of_layers", "double",
                "--momentum", "0.6",
                "--lr", "0.05",
                "--r", str(r)
            ]

    def run_experiment(self, cmd):
        """Run a single experiment with GPU cleanup"""
        self._clear_gpu_memory()
        
        print(f"\n▶ Running: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                timeout=14400  # 4 hour timeout
            )
            print(f"✓ Completed in {time.time()-start_time:.1f}s")
            return True
            
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout after {time.time()-start_time:.1f}s")
            return False
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed with code {e.returncode}")
            print(f"Error: {e.stderr[:500]}...")  # Show first 500 chars of error
            return False

    def run_all(self):
        """Run all experiments sequentially"""
        total = 0
        success = 0
        
        for cmd in self.generate_commands():
            total += 1
            if self.run_experiment(cmd):
                success += 1
            time.sleep(1)  # Brief pause between experiments
        
        print(f"\n🎉 Completed {success}/{total} experiments successfully")

if __name__ == "__main__":
    runner = GPUExperimentRunner()
    runner.run_all()