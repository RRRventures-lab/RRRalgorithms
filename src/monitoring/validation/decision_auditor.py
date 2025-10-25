from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import json
import logging
import time

#!/usr/bin/env python3

"""
Decision Auditor - Immutable Audit Trail System

Creates cryptographically-verified immutable audit logs for all AI trading decisions.
Enables complete traceability, forensic analysis, and regulatory compliance.

Features:
- Immutable append-only log
- Cryptographic hash chaining (blockchain-style)
- Complete decision context capture
- Tamper detection
- Audit trail verification

Author: AI Psychology Team
Date: 2025-10-11
"""


logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of decisions to audit"""
    TRADE = "trade"
    POSITION_SIZING = "position_sizing"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    RISK_LIMIT = "risk_limit"
    EMERGENCY_STOP = "emergency_stop"


class ValidationStatus(Enum):
    """Validation status"""
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"
    WARNING = "warning"


@dataclass
class AuditEntry:
    """
    Immutable audit entry for a single decision

    Each entry is cryptographically linked to previous entry via hash chaining
    """
    # Entry metadata
    entry_id: str
    timestamp: datetime
    previous_hash: str

    # Decision details
    decision_id: str
    decision_type: DecisionType
    model_version: str
    model_name: str

    # Input context
    inputs: Dict[str, Any]
    market_context: Dict[str, Any]

    # Output decision
    output: Dict[str, Any]
    confidence: float

    # Reasoning and explainability
    reasoning: List[str]
    feature_importance: Dict[str, float]

    # Validation results
    validation_status: ValidationStatus
    hallucination_reports: List[Dict[str, Any]]
    data_authenticity_score: float
    decision_logic_valid: bool

    # Risk assessment
    risk_metrics: Dict[str, float]
    expected_value: float
    max_loss: float
    probability_success: float

    # Data sources
    data_sources: List[Dict[str, Any]]

    # Execution results (filled in later)
    execution_timestamp: Optional[datetime] = None
    execution_status: Optional[str] = None
    actual_outcome: Optional[Dict[str, Any]] = None

    # Cryptographic verification
    entry_hash: str = field(default="", init=False)

    def __post_init__(self):
        """Calculate entry hash after initialization"""
        if not self.entry_hash:
            self.entry_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """
        Calculate SHA-256 hash of entry

        Hash includes all entry data plus previous hash for chain verification
        """
        # Create canonical representation
        entry_dict = asdict(self)

        # Remove hash field itself
        entry_dict.pop('entry_hash', None)

        # Convert datetime to ISO format
        entry_dict['timestamp'] = self.timestamp.isoformat()
        if self.execution_timestamp:
            entry_dict['execution_timestamp'] = self.execution_timestamp.isoformat()

        # Convert enums to strings
        entry_dict['decision_type'] = self.decision_type.value
        entry_dict['validation_status'] = self.validation_status.value

        # Sort keys for deterministic hashing
        canonical_json = json.dumps(entry_dict, sort_keys=True)

        # Calculate SHA-256 hash
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def verify_hash(self) -> bool:
        """Verify that stored hash matches calculated hash"""
        return self.entry_hash == self._calculate_hash()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        if self.execution_timestamp:
            d['execution_timestamp'] = self.execution_timestamp.isoformat()
        d['decision_type'] = self.decision_type.value
        d['validation_status'] = self.validation_status.value
        return d


class DecisionAuditor:
    """
    Immutable audit trail system for AI decisions

    Maintains append-only log with cryptographic hash chaining
    """

    def __init__(
        self,
        audit_log_path: str = "/var/log/rrr-trading/audit/decisions.jsonl",
        enable_file_logging: bool = True,
        enable_database_logging: bool = False,
        database_url: Optional[str] = None
    ):
        self.audit_log_path = Path(audit_log_path)
        self.enable_file_logging = enable_file_logging
        self.enable_database_logging = enable_database_logging
        self.database_url = database_url

        # In-memory cache of recent entries
        self.recent_entries: List[AuditEntry] = []
        self.max_cache_size = 1000

        # Last hash in chain
        self.last_hash: str = "0" * 64  # Genesis hash

        # Statistics
        self.total_entries = 0
        self.entries_by_type: Dict[DecisionType, int] = {}
        self.entries_by_status: Dict[ValidationStatus, int] = {}

        # Initialize audit log
        self._initialize_audit_log()

        logger.info(f"DecisionAuditor initialized with log path: {self.audit_log_path}")

    def _initialize_audit_log(self):
        """Initialize audit log file and load last hash"""
        if self.enable_file_logging:
            # Create directory if it doesn't exist
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

            # Load last hash if log exists
            if self.audit_log_path.exists():
                self.last_hash = self._load_last_hash()
                logger.info(f"Loaded existing audit log, last hash: {self.last_hash[:16]}...")
            else:
                # Create genesis entry
                self._create_genesis_entry()

    def _create_genesis_entry(self):
        """Create genesis entry (first entry in chain)"""
        genesis_entry = {
            "entry_id": "genesis",
            "timestamp": datetime.utcnow().isoformat(),
            "previous_hash": "0" * 64,
            "entry_type": "genesis",
            "description": "RRR Trading System Audit Trail Genesis Block",
            "version": "1.0.0"
        }

        genesis_hash = hashlib.sha256(
            json.dumps(genesis_entry, sort_keys=True).encode('utf-8')
        ).hexdigest()

        genesis_entry['entry_hash'] = genesis_hash

        # Write genesis entry
        with open(self.audit_log_path, 'w') as f:
            f.write(json.dumps(genesis_entry) + '\n')

        self.last_hash = genesis_hash
        logger.info(f"Created genesis entry with hash: {genesis_hash[:16]}...")

    def _load_last_hash(self) -> str:
        """Load last hash from audit log"""
        try:
            with open(self.audit_log_path, 'r') as f:
                # Read last line
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    return last_entry.get('entry_hash', "0" * 64)
        except Exception as e:
            logger.error(f"Error loading last hash: {e}")

        return "0" * 64

    def log_decision(
        self,
        decision_id: str,
        decision_type: DecisionType,
        model_version: str,
        model_name: str,
        inputs: Dict[str, Any],
        market_context: Dict[str, Any],
        output: Dict[str, Any],
        confidence: float,
        reasoning: List[str],
        feature_importance: Dict[str, float],
        validation_status: ValidationStatus,
        hallucination_reports: List[Dict[str, Any]],
        data_authenticity_score: float,
        decision_logic_valid: bool,
        risk_metrics: Dict[str, float],
        expected_value: float,
        max_loss: float,
        probability_success: float,
        data_sources: List[Dict[str, Any]]
    ) -> AuditEntry:
        """
        Log a decision to the immutable audit trail

        Returns:
            AuditEntry with calculated hash
        """
        # Create entry
        entry = AuditEntry(
            entry_id=self._generate_entry_id(),
            timestamp=datetime.utcnow(),
            previous_hash=self.last_hash,
            decision_id=decision_id,
            decision_type=decision_type,
            model_version=model_version,
            model_name=model_name,
            inputs=inputs,
            market_context=market_context,
            output=output,
            confidence=confidence,
            reasoning=reasoning,
            feature_importance=feature_importance,
            validation_status=validation_status,
            hallucination_reports=hallucination_reports,
            data_authenticity_score=data_authenticity_score,
            decision_logic_valid=decision_logic_valid,
            risk_metrics=risk_metrics,
            expected_value=expected_value,
            max_loss=max_loss,
            probability_success=probability_success,
            data_sources=data_sources
        )

        # Verify entry hash is calculated correctly
        if not entry.verify_hash():
            logger.error("Entry hash verification failed!")
            raise ValueError("Entry hash verification failed")

        # Persist entry
        if self.enable_file_logging:
            self._write_entry_to_file(entry)

        if self.enable_database_logging:
            self._write_entry_to_database(entry)

        # Update last hash
        self.last_hash = entry.entry_hash

        # Add to cache
        self.recent_entries.append(entry)
        if len(self.recent_entries) > self.max_cache_size:
            self.recent_entries.pop(0)

        # Update statistics
        self.total_entries += 1
        self.entries_by_type[decision_type] = self.entries_by_type.get(decision_type, 0) + 1
        self.entries_by_status[validation_status] = self.entries_by_status.get(validation_status, 0) + 1

        logger.info(f"Logged decision {decision_id} with hash {entry.entry_hash[:16]}...")

        return entry

    def update_execution_result(
        self,
        decision_id: str,
        execution_status: str,
        actual_outcome: Dict[str, Any]
    ) -> bool:
        """
        Update execution result for a logged decision

        Note: This creates a NEW entry linking to the original decision,
        maintaining immutability of original entry.
        """
        # Create execution update entry
        update_entry = {
            "entry_id": self._generate_entry_id(),
            "timestamp": datetime.utcnow().isoformat(),
            "previous_hash": self.last_hash,
            "entry_type": "execution_update",
            "decision_id": decision_id,
            "execution_timestamp": datetime.utcnow().isoformat(),
            "execution_status": execution_status,
            "actual_outcome": actual_outcome
        }

        # Calculate hash
        update_hash = hashlib.sha256(
            json.dumps(update_entry, sort_keys=True).encode('utf-8')
        ).hexdigest()
        update_entry['entry_hash'] = update_hash

        # Write to log
        if self.enable_file_logging:
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(update_entry) + '\n')

        # Update last hash
        self.last_hash = update_hash

        logger.info(f"Logged execution update for decision {decision_id}")

        return True

    def verify_chain_integrity(self, start_entry_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify integrity of entire audit chain

        Checks:
        - Hash chain is unbroken
        - Each entry's hash matches its content
        - No entries have been tampered with

        Returns:
            Verification report
        """
        if not self.audit_log_path.exists():
            return {
                "verified": False,
                "error": "Audit log does not exist"
            }

        entries_verified = 0
        entries_failed = 0
        broken_chain = False
        tampered_entries = []

        previous_hash = None

        with open(self.audit_log_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry_dict = json.loads(line)

                    # Check hash chain
                    if previous_hash is not None:
                        if entry_dict.get('previous_hash') != previous_hash:
                            broken_chain = True
                            logger.error(f"Broken chain at line {line_num}: expected {previous_hash[:16]}, got {entry_dict.get('previous_hash', '')[:16]}")

                    # Verify entry hash
                    stored_hash = entry_dict.get('entry_hash')
                    # Recalculate hash
                    entry_dict_copy = entry_dict.copy()
                    entry_dict_copy.pop('entry_hash', None)
                    calculated_hash = hashlib.sha256(
                        json.dumps(entry_dict_copy, sort_keys=True).encode('utf-8')
                    ).hexdigest()

                    if stored_hash != calculated_hash:
                        tampered_entries.append({
                            "line_num": line_num,
                            "entry_id": entry_dict.get('entry_id'),
                            "stored_hash": stored_hash,
                            "calculated_hash": calculated_hash
                        })
                        entries_failed += 1
                    else:
                        entries_verified += 1

                    previous_hash = stored_hash

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON at line {line_num}: {e}")
                    entries_failed += 1

        verified = not broken_chain and entries_failed == 0

        report = {
            "verified": verified,
            "entries_verified": entries_verified,
            "entries_failed": entries_failed,
            "broken_chain": broken_chain,
            "tampered_entries": tampered_entries,
            "last_verified_hash": previous_hash
        }

        if verified:
            logger.info(f"Chain integrity verified: {entries_verified} entries OK")
        else:
            logger.error(f"Chain integrity FAILED: {entries_failed} entries failed, broken_chain={broken_chain}")

        return report

    def _generate_entry_id(self) -> str:
        """Generate unique entry ID"""
        timestamp = datetime.utcnow().isoformat()
        nonce = str(time.time_ns())
        return hashlib.sha256(f"{timestamp}:{nonce}".encode('utf-8')).hexdigest()[:16]

    def _write_entry_to_file(self, entry: AuditEntry):
        """Write entry to file (append-only)"""
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')

    def _write_entry_to_database(self, entry: AuditEntry):
        """Write entry to database (future implementation)"""
        # TODO: Implement database logging
        pass

    def get_decision_history(self, decision_id: str) -> List[Dict[str, Any]]:
        """Get complete history for a decision ID"""
        history = []

        if self.audit_log_path.exists():
            with open(self.audit_log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get('decision_id') == decision_id:
                        history.append(entry)

        return history

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics"""
        return {
            "total_entries": self.total_entries,
            "entries_by_type": {k.value: v for k, v in self.entries_by_type.items()},
            "entries_by_status": {k.value: v for k, v in self.entries_by_status.items()},
            "last_hash": self.last_hash[:16] + "...",
            "audit_log_path": str(self.audit_log_path)
        }
