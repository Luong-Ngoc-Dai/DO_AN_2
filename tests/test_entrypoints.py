"""Exercise actual module/root CLI startup without launching training."""
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys


def test_all_cli_help():
    commands = [[sys.executable, '-m', f'src.{module}', '--help']
                for module in ('dataset','model','train','evaluate','inference','compare')]
    commands += [[sys.executable, f'{module}.py', '--help']
                 for module in ('dataset','model','train','evaluate','inference')]
    def check(command):
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, f'{command}: {result.stderr}'
        assert 'usage:' in result.stdout, f'{command}: missing CLI help'
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(check, commands))
