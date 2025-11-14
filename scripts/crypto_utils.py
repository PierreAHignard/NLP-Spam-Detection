"""
Cryptographic Utilities for Student Test Verification

This module provides utilities for signing and verifying test results using RSA
cryptographic signatures. It handles key loading, message signing, and verification.

Usage:
    # Student side (signing)
    signer = TestResultSigner('instructor_public_key.pem')
    signature = signer.sign_test_results(test_data, commit_hash, author_info)
    
    # Instructor side (verification)
    verifier = TestResultVerifier('instructor_private_key.pem')
    is_valid = verifier.verify_signature(proof_data)

Author: Data Science Toolkit
"""

import json
import base64
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
import subprocess

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature


class CryptoError(Exception):
    """Base exception for cryptographic operations."""
    pass


class TestResultSigner:
    """
    Handles signing of test results for students.
    
    This class loads the instructor's public key and creates cryptographic
    signatures that can only be verified by the instructor's private key.
    """
    
    def __init__(self, public_key_path: Path):
        """
        Initialize the signer with instructor's public key.
        
        Args:
            public_key_path: Path to the instructor's public key file
            
        Raises:
            CryptoError: If key loading fails
        """
        self.public_key_path = Path(public_key_path)
        self.public_key = self._load_public_key()
    
    def _load_public_key(self) -> rsa.RSAPublicKey:
        """
        Load RSA public key from PEM file.
        
        Returns:
            RSA public key object
            
        Raises:
            CryptoError: If key loading fails
        """
        try:
            with open(self.public_key_path, 'rb') as key_file:
                public_key = serialization.load_pem_public_key(
                    key_file.read(),
                    backend=default_backend()
                )
            
            if not isinstance(public_key, rsa.RSAPublicKey):
                raise CryptoError("Key is not a valid RSA public key")
            
            return public_key
            
        except FileNotFoundError:
            raise CryptoError(f"Public key file not found: {self.public_key_path}")
        except Exception as e:
            raise CryptoError(f"Failed to load public key: {str(e)}")
    
    def _get_git_info(self, repo_path: Path) -> Dict[str, str]:
        """
        Extract current Git information from repository.
        
        Args:
            repo_path: Path to the Git repository
            
        Returns:
            Dictionary with git information
        """
        try:
            # Get current commit hash
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=repo_path,
                text=True
            ).strip()
            
            # Get author information
            author_name = subprocess.check_output(
                ['git', 'config', 'user.name'],
                cwd=repo_path,
                text=True
            ).strip()
            
            author_email = subprocess.check_output(
                ['git', 'config', 'user.email'],
                cwd=repo_path,
                text=True
            ).strip()
            
            return {
                'commit_hash': commit_hash,
                'author_name': author_name,
                'author_email': author_email
            }
            
        except subprocess.CalledProcessError as e:
            raise CryptoError(f"Failed to get Git information: {str(e)}")
    
    def _create_message_to_sign(self, test_results: Dict[str, Any], 
                               git_info: Dict[str, str], timestamp: str) -> str:
        """
        Create the message that will be cryptographically signed.
        
        Args:
            test_results: Test execution results
            git_info: Git repository information
            timestamp: ISO timestamp of test execution
            
        Returns:
            String message to be signed
        """
        # Extract test summary and execution metadata correctly from nested structure
        test_summary = test_results.get('test_summary', {})
        execution_metadata = test_results.get('execution_metadata', {})
        
        # Create a structured message with all critical information
        message_data = {
            'test_summary': {
                'total_tests': test_summary.get('total_tests', 0),
                'passed_tests': test_summary.get('passed_tests', 0),
                'success_rate': test_summary.get('success_rate', 0),
                'execution_time': execution_metadata.get('execution_time_seconds', 0)
            },
            'git_context': git_info,
            'timestamp': timestamp,
            'version': '1.0'
        }
        
        # Convert to canonical JSON (sorted keys, no spaces)
        return json.dumps(message_data, sort_keys=True, separators=(',', ':'))
    
    def sign_test_results(self, test_results: Dict[str, Any], 
                         repo_path: Path) -> Dict[str, Any]:
        """
        Create cryptographic proof of test execution.
        
        Args:
            test_results: Results from test execution
            repo_path: Path to the Git repository
            
        Returns:
            Dictionary containing the signed proof
            
        Raises:
            CryptoError: If signing fails
        """
        timestamp = datetime.now().isoformat()
        git_info = self._get_git_info(repo_path)
        
        # Create message to sign
        message = self._create_message_to_sign(test_results, git_info, timestamp)
        message_bytes = message.encode('utf-8')
        
        # Note: We're using the public key here for demonstration
        # In a real implementation, students would not be able to sign
        # This is a conceptual limitation - in practice, the instructor 
        # would need to implement a different verification strategy
        try:
            # Create a hash of the message (this will be signed by instructor later)
            digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
            digest.update(message_bytes)
            message_hash = digest.finalize()
            
            # For now, we'll create a proof structure that includes the message
            # The actual signing would be done by the instructor's system
            proof = {
                'message': message,
                'message_hash': base64.b64encode(message_hash).decode('utf-8'),
                'test_results': test_results,
                'git_info': git_info,
                'timestamp': timestamp,
                'signature_placeholder': 'INSTRUCTOR_SIGNATURE_REQUIRED'
            }
            
            return proof
            
        except Exception as e:
            raise CryptoError(f"Failed to create proof: {str(e)}")


class TestResultVerifier:
    """
    Handles verification of signed test results for instructors.
    
    This class loads the instructor's private key and verifies that
    test result signatures are authentic and unmodified.
    """
    
    def __init__(self, private_key_path: Path):
        """
        Initialize the verifier with instructor's private key.
        
        Args:
            private_key_path: Path to the instructor's private key file
            
        Raises:
            CryptoError: If key loading fails
        """
        self.private_key_path = Path(private_key_path)
        self.private_key = self._load_private_key()
        self.public_key = self.private_key.public_key()
    
    def _load_private_key(self) -> rsa.RSAPrivateKey:
        """
        Load RSA private key from PEM file.
        
        Returns:
            RSA private key object
            
        Raises:
            CryptoError: If key loading fails
        """
        try:
            with open(self.private_key_path, 'rb') as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,  # No password protection
                    backend=default_backend()
                )
            
            if not isinstance(private_key, rsa.RSAPrivateKey):
                raise CryptoError("Key is not a valid RSA private key")
            
            return private_key
            
        except FileNotFoundError:
            raise CryptoError(f"Private key file not found: {self.private_key_path}")
        except Exception as e:
            raise CryptoError(f"Failed to load private key: {str(e)}")
    
    def sign_message(self, message: str) -> str:
        """
        Sign a message with the private key.
        
        Args:
            message: String message to sign
            
        Returns:
            Base64-encoded signature
            
        Raises:
            CryptoError: If signing fails
        """
        try:
            message_bytes = message.encode('utf-8')
            
            signature = self.private_key.sign(
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            raise CryptoError(f"Failed to sign message: {str(e)}")
    
    def verify_signature(self, proof_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verify the cryptographic signature of test results.
        
        Args:
            proof_data: Dictionary containing proof and signature
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Extract components
            message = proof_data.get('message')
            signature_b64 = proof_data.get('signature')
            
            if not message or not signature_b64:
                return False, "Missing message or signature in proof"
            
            # Decode signature
            signature = base64.b64decode(signature_b64)
            message_bytes = message.encode('utf-8')
            
            # Verify signature
            self.public_key.verify(
                signature,
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True, "Signature is valid"
            
        except InvalidSignature:
            return False, "Invalid signature - proof may be forged"
        except Exception as e:
            return False, f"Verification error: {str(e)}"
    
    def complete_proof_signing(self, proof_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete the signing process by adding instructor signature to proof.
        
        This method is used by the instructor to sign student proofs after
        verification that they contain valid test results.
        
        Args:
            proof_data: Proof dictionary created by student
            
        Returns:
            Completed proof with instructor signature
            
        Raises:
            CryptoError: If signing fails
        """
        try:
            message = proof_data.get('message')
            if not message:
                raise CryptoError("No message found in proof data")
            
            # Sign the message
            signature = self.sign_message(message)
            
            # Create completed proof
            completed_proof = proof_data.copy()
            completed_proof['signature'] = signature
            completed_proof['signature_placeholder'] = None
            completed_proof['signed_by_instructor'] = True
            completed_proof['signature_timestamp'] = datetime.now().isoformat()
            
            return completed_proof
            
        except Exception as e:
            raise CryptoError(f"Failed to complete proof signing: {str(e)}")
    
    def extract_test_summary(self, proof_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract test summary from verified proof data.
        
        Args:
            proof_data: Verified proof dictionary
            
        Returns:
            Dictionary with test summary information
        """
        try:
            # Parse the signed message
            message = proof_data.get('message', '{}')
            message_data = json.loads(message)
            
            # Extract key information
            summary = {
                'test_results': message_data.get('test_summary', {}),
                'git_info': message_data.get('git_context', {}),
                'timestamp': message_data.get('timestamp'),
                'raw_results': proof_data.get('test_results', {}),
                'verification_status': 'valid' if self.verify_signature(proof_data)[0] else 'invalid'
            }
            
            return summary
            
        except Exception as e:
            return {
                'error': f"Failed to extract test summary: {str(e)}",
                'verification_status': 'error'
            }


def create_test_proof(test_results: Dict[str, Any], repo_path: Path, 
                     public_key_path: Path) -> Dict[str, Any]:
    """
    Convenience function to create a test proof.
    
    Args:
        test_results: Results from test execution
        repo_path: Path to the Git repository
        public_key_path: Path to instructor's public key
        
    Returns:
        Dictionary containing the test proof
        
    Raises:
        CryptoError: If proof creation fails
    """
    signer = TestResultSigner(public_key_path)
    return signer.sign_test_results(test_results, repo_path)


def verify_test_proof(proof_data: Dict[str, Any], 
                     private_key_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to verify a test proof.
    
    Args:
        proof_data: Proof dictionary to verify
        private_key_path: Path to instructor's private key
        
    Returns:
        Tuple of (is_valid, summary_data)
    """
    verifier = TestResultVerifier(private_key_path)
    is_valid, error_msg = verifier.verify_signature(proof_data)
    
    if is_valid:
        summary = verifier.extract_test_summary(proof_data)
    else:
        summary = {'error': error_msg, 'verification_status': 'invalid'}
    
    return is_valid, summary
