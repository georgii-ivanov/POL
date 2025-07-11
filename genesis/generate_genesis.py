#!/usr/bin/env python3

import os
import sys
import hashlib
import json
from typing import Dict, Any
import secrets


def derive_bls_keys(seed: str) -> tuple:
    seed_bytes = bytes.fromhex(seed)
    private_key_hash = hashlib.sha256(seed_bytes + b"private_key").digest()
    private_key = int.from_bytes(private_key_hash, "big") % (2 ** 255)
    public_key_hash = hashlib.sha256(seed_bytes + b"public_key").digest()
    public_key = public_key_hash + hashlib.sha256(public_key_hash).digest()[:16]
    withdrawal_hash = hashlib.sha256(seed_bytes + b"withdrawal").digest()
    withdrawal_credentials = b"\x00" + withdrawal_hash[:31]
    return private_key.to_bytes(32, "big"), public_key, withdrawal_credentials


def create_genesis_validator(seed: str) -> tuple:
    private_key, public_key, withdrawal_credentials = derive_bls_keys(seed)
    validator = {
        "pubkey": "0x" + public_key.hex(),
        "withdrawal_credentials": "0x" + withdrawal_credentials.hex(),
        "effective_balance": "32000000000",
        "slashed": False,
        "activation_eligibility_epoch": "0",
        "activation_epoch": "0",
        "exit_epoch": "18446744073709551615",
        "withdrawable_epoch": "18446744073709551615",
    }
    return validator, private_key


def create_genesis_state(seed: str) -> tuple:
    validator, private_key = create_genesis_validator(seed)
    genesis_time = 1640995200
    state = {
        "genesis_time": str(genesis_time),
        "genesis_validators_root": "0x" + ("0" * 64),
        "slot": "0",
        "fork": {
            "previous_version": "0x20000000",
            "current_version": "0x20000000",
            "epoch": "0",
        },
        "latest_block_header": {
            "slot": "0",
            "proposer_index": "0",
            "parent_root": "0x" + ("0" * 64),
            "state_root": "0x" + ("0" * 64),
            "body_root": "0x" + ("0" * 64),
        },
        "block_roots": ["0x" + ("0" * 64)] * 8192,
        "state_roots": ["0x" + ("0" * 64)] * 8192,
        "historical_roots": [],
        "eth1_data": {
            "deposit_root": "0x" + ("0" * 64),
            "deposit_count": "1",
            "block_hash": "0x" + ("0" * 64),
        },
        "eth1_data_votes": [],
        "eth1_deposit_index": "1",
        "validators": [validator],
        "balances": ["32000000000"],
        "randao_mixes": ["0x" + ("0" * 64)] * 65536,
        "slashings": ["0"] * 8192,
        "previous_epoch_participation": ["0xff"],
        "current_epoch_participation": ["0xff"],
        "justification_bits": "0x00",
        "previous_justified_checkpoint": {"epoch": "0", "root": "0x" + ("0" * 64)},
        "current_justified_checkpoint": {"epoch": "0", "root": "0x" + ("0" * 64)},
        "finalized_checkpoint": {"epoch": "0", "root": "0x" + ("0" * 64)},
        "inactivity_scores": ["0"],
        "current_sync_committee": {
            "pubkeys": [validator["pubkey"]] * 512,
            "aggregate_pubkey": validator["pubkey"],
        },
        "next_sync_committee": {
            "pubkeys": [validator["pubkey"]] * 512,
            "aggregate_pubkey": validator["pubkey"],
        },
        "latest_execution_payload_header": {
            "parent_hash": "0x" + ("0" * 64),
            "fee_recipient": "0x" + ("0" * 40),
            "state_root": "0x" + ("0" * 64),
            "receipts_root": "0x" + ("0" * 64),
            "logs_bloom": "0x" + ("0" * 512),
            "prev_randao": "0x" + ("0" * 64),
            "block_number": "0",
            "gas_limit": "30000000",
            "gas_used": "0",
            "timestamp": str(genesis_time),
            "extra_data": "0x",
            "base_fee_per_gas": "1000000000",
            "block_hash": "0x" + ("0" * 64),
            "transactions_root": "0x" + ("0" * 64),
            "withdrawals_root": "0x" + ("0" * 64),
        },
    }
    return state, private_key


def save_genesis_files(seed: str, output_dir: str = "/data"):
    os.makedirs(output_dir, exist_ok=True)
    genesis_state, private_key = create_genesis_state(seed)
    with open(f"{output_dir}/genesis.json", "w") as f:
        json.dump(genesis_state, f, indent=2)
    with open(f"{output_dir}/validator_private_key.hex", "w") as f:
        f.write(private_key.hex())
    keystore = {
        "crypto": {
            "kdf": {
                "function": "pbkdf2",
                "params": {"dklen": 32, "n": 262144, "r": 8, "p": 1, "salt": secrets.token_hex(32)},
                "message": "",
            },
            "checksum": {"function": "sha256", "params": {}, "message": ""},
            "cipher": {"function": "aes-128-ctr", "params": {"iv": secrets.token_hex(16)}, "message": private_key.hex()},
        },
        "description": "POL-AI Genesis Validator",
        "pubkey": genesis_state["validators"][0]["pubkey"][2:],
        "path": "m/12381/3600/0/0/0",
        "uuid": "00000000-0000-0000-0000-000000000001",
        "version": 4,
    }
    with open(f"{output_dir}/keystore.json", "w") as f:
        json.dump(keystore, f, indent=2)


def main():
    seed = os.getenv("GENESIS_SEED") or secrets.token_hex(32)
    try:
        bytes.fromhex(seed)
    except ValueError:
        sys.exit(1)
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/data"
    save_genesis_files(seed, output_dir)


if __name__ == "__main__":
    main() 