"""CPU evaluation tests; suppress bytecode outside the authorized file scope."""

import sys

sys.dont_write_bytecode = True
