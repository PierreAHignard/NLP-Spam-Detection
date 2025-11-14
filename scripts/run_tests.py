#!/usr/bin/env python3
"""
Test Runner Script for Air Quality ML Pipeline with Cryptographic Proof

This script provides a comprehensive test runner for validating the technical
implementation of the Air Quality ML Pipeline components, now enhanced with
cryptographic proof generation for instructor verification.
"""

import sys
import argparse
import os
import json
import time
import platform
from pathlib import Path
import subprocess
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Test imports
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

# Debug: Check if cryptography is available
try:
    import cryptography
    CRYPTOGRAPHY_LIB_AVAILABLE = True
    if '--verbose' in sys.argv or '-v' in sys.argv:
        print(f"🔍 DEBUG: cryptography library found at {cryptography.__file__}")
except ImportError:
    CRYPTOGRAPHY_LIB_AVAILABLE = False
    if '--verbose' in sys.argv or '-v' in sys.argv:
        print("🔍 DEBUG: cryptography library not found")

# Import crypto utilities with comprehensive error handling
CRYPTO_AVAILABLE = False
CryptoError = Exception  # Fallback
create_test_proof = None

if CRYPTOGRAPHY_LIB_AVAILABLE:
    # Try different import paths
    import_attempts = []
    
    try:
        # Attempt 1: Direct import from current directory
        from crypto_utils import create_test_proof, CryptoError
        CRYPTO_AVAILABLE = True
        import_attempts.append("✅ Direct import from crypto_utils")
    except ImportError as e:
        import_attempts.append(f"❌ Direct import failed: {e}")
        
        try:
            # Attempt 2: Import from scripts directory
            from scripts.crypto_utils import create_test_proof, CryptoError
            CRYPTO_AVAILABLE = True
            import_attempts.append("✅ Import from scripts.crypto_utils")
        except ImportError as e:
            import_attempts.append(f"❌ scripts.crypto_utils failed: {e}")
            
            try:
                # Attempt 3: Add scripts to path and import
                sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
                from crypto_utils import create_test_proof, CryptoError
                CRYPTO_AVAILABLE = True
                import_attempts.append("✅ Import after adding scripts to path")
            except ImportError as e:
                import_attempts.append(f"❌ Path addition failed: {e}")
                
                # Attempt 4: Try to create minimal crypto functions inline
                try:
                    from cryptography.hazmat.primitives import hashes, serialization
                    from cryptography.hazmat.primitives.asymmetric import rsa, padding
                    from cryptography.hazmat.backends import default_backend
                    import base64
                    
                    # Create minimal inline crypto functions
                    class CryptoError(Exception):
                        pass
                    
                    def create_test_proof(test_results, repo_path, public_key_path):
                        """Minimal inline proof creation."""
                        timestamp = datetime.now().isoformat()
                        
                        try:
                            # Get git info
                            commit_hash = subprocess.check_output(
                                ['git', 'rev-parse', 'HEAD'], cwd=repo_path, text=True
                            ).strip()
                            author_name = subprocess.check_output(
                                ['git', 'config', 'user.name'], cwd=repo_path, text=True
                            ).strip()
                            author_email = subprocess.check_output(
                                ['git', 'config', 'user.email'], cwd=repo_path, text=True
                            ).strip()
                        except subprocess.CalledProcessError:
                            commit_hash = "unknown"
                            author_name = "unknown"
                            author_email = "unknown"
                        
                        # Create proof structure
                        message_data = {
                            'test_summary': {
                                'total_tests': test_results.get('test_summary', {}).get('total_tests', 0),
                                'passed_tests': test_results.get('test_summary', {}).get('passed_tests', 0),
                                'success_rate': test_results.get('test_summary', {}).get('success_rate', 0)
                            },
                            'git_context': {
                                'commit_hash': commit_hash,
                                'author_name': author_name,
                                'author_email': author_email
                            },
                            'timestamp': timestamp,
                            'version': '1.0'
                        }
                        
                        message = json.dumps(message_data, sort_keys=True, separators=(',', ':'))
                        
                        # Create hash for instructor signing
                        digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
                        digest.update(message.encode('utf-8'))
                        message_hash = digest.finalize()
                        
                        return {
                            'message': message,
                            'message_hash': base64.b64encode(message_hash).decode('utf-8'),
                            'test_results': test_results,
                            'git_info': {
                                'commit_hash': commit_hash,
                                'author_name': author_name,
                                'author_email': author_email
                            },
                            'timestamp': timestamp,
                            'signature_placeholder': 'INSTRUCTOR_SIGNATURE_REQUIRED'
                        }
                    
                    CRYPTO_AVAILABLE = True
                    import_attempts.append("✅ Inline crypto functions created")
                    
                except ImportError as e:
                    import_attempts.append(f"❌ Inline crypto creation failed: {e}")

    # Debug output for verbose mode
    if ('--verbose' in sys.argv or '-v' in sys.argv) and import_attempts:
        print("🔍 DEBUG: Crypto import attempts:")
        for attempt in import_attempts:
            print(f"    {attempt}")


class TestRunner:
    """Professional test runner for the Air Quality ML Pipeline with crypto support."""
    
    def __init__(self, verbose=False, generate_proof=True):
        """
        Initialize the test runner.
        
        Args:
            verbose: Enable verbose output
            generate_proof: Whether to generate cryptographic proof
        """
        self.verbose = verbose
        self.project_root = PROJECT_ROOT
        self.tests_dir = self.project_root / "tests"
        self.start_time = None
        self.results = {}
        
        # Crypto settings
        self.generate_proof = generate_proof and CRYPTO_AVAILABLE
        self.public_key_path = self.project_root / "instructor_public_key.pem"
        self.proof_output_path = self.project_root / "test_proof.json"
        self.results_output_path = self.project_root / "test_results.json"
        
        # Debug info
        if verbose:
            print(f"🔍 DEBUG: CRYPTOGRAPHY_LIB_AVAILABLE = {CRYPTOGRAPHY_LIB_AVAILABLE}")
            print(f"🔍 DEBUG: CRYPTO_AVAILABLE = {CRYPTO_AVAILABLE}")
            print(f"🔍 DEBUG: generate_proof = {self.generate_proof}")
            print(f"🔍 DEBUG: public_key_path = {self.public_key_path}")
            print(f"🔍 DEBUG: public_key_exists = {self.public_key_path.exists()}")
    
    def print_header(self, title):
        """Print a formatted header."""
        if self.verbose:
            separator = "=" * 60
            print(f"\n{separator}")
            print(f"🧪 {title.upper()}")
            print(separator)
        else:
            print(f"\n🧪 {title}")
    
    def print_step(self, message):
        """Print a step message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.verbose:
            print(f"[{timestamp}] ├── {message}")
        else:
            print(f"├── {message}")
    
    def print_success(self, message):
        """Print a success message."""
        print(f"✅ {message}")
    
    def print_error(self, message):
        """Print an error message."""
        print(f"❌ {message}")
    
    def print_warning(self, message):
        """Print a warning message."""
        print(f"⚠️  {message}")
    
    def check_dependencies(self):
        """Check if required testing dependencies are available."""
        self.print_step("Checking dependencies...")
        
        missing_deps = []
        
        if not PYTEST_AVAILABLE:
            missing_deps.append("pytest")
        
        if missing_deps:
            self.print_error(f"Missing dependencies: {', '.join(missing_deps)}")
            self.print_step("Install with: uv add --dev pytest")
            return False
        
        self.print_success("All dependencies available")
        return True
    
    def check_project_structure(self):
        """Check if project structure is correct."""
        self.print_step("Checking project structure...")
        
        required_paths = [
            self.project_root / "src",
            self.project_root / "src" / "pipeline",
            self.project_root / "src" / "utils",
            self.tests_dir,
        ]
        
        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(str(path))
        
        if missing_paths:
            self.print_error(f"Missing directories: {missing_paths}")
            return False
        
        self.print_success("Project structure OK")
        return True
    
    def check_crypto_setup(self):
        """Check if cryptographic setup is available and valid."""
        if not self.generate_proof:
            return True  # No crypto needed
        
        self.print_step("Checking cryptographic setup...")
        
        if not CRYPTOGRAPHY_LIB_AVAILABLE:
            self.print_warning("Cryptography library not found")
            self.print_step("Install with: uv add cryptography")
            return False
        
        if not CRYPTO_AVAILABLE:
            self.print_warning("Crypto utilities not available")
            self.print_step("Missing crypto_utils.py module")
            if self.verbose:
                self.print_step("Make sure crypto_utils.py is in the scripts/ directory")
            return False
        
        if not self.public_key_path.exists():
            self.print_warning(f"Instructor public key not found: {self.public_key_path}")
            self.print_step("Contact your instructor for the public key file")
            return False
        
        self.print_success("Cryptographic setup verified")
        return True
    
    def check_git_status(self):
        """Enhanced git status check with detailed information."""
        self.print_step("Checking git status and commit information...")
        
        try:
            # Check if we're in a git repository
            subprocess.check_output(
                ['git', 'rev-parse', '--git-dir'], 
                cwd=self.project_root,
                stderr=subprocess.DEVNULL
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.print_warning("Not a git repository - proof will be limited")
            return True
        
        try:
            # Check for uncommitted changes
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'], 
                cwd=self.project_root
            ).decode().strip()
            
            if status:
                self.print_warning("Uncommitted changes detected:")
                for line in status.splitlines()[:5]:  # Show max 5 files
                    self.print_step(f"  {line}")
                if len(status.splitlines()) > 5:
                    self.print_step(f"  ... and {len(status.splitlines()) - 5} more files")
                
                # Allow tests to run but warn about proof implications
                self.print_step("Tests will run, but proof will reflect current commit state")
                return True
            
            # Get current commit info
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                cwd=self.project_root
            ).decode().strip()
            
            author_name = subprocess.check_output(
                ['git', 'config', 'user.name'], 
                cwd=self.project_root
            ).decode().strip()
            
            self.print_success(f"Git status clean - Commit: {commit_hash[:8]}, Author: {author_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.print_warning(f"Git status check failed: {e}")
            return True  # Allow tests to run
    
    def run_pytest_module(self, module_name):
        """
        Run tests for a specific module using pytest.
        
        Args:
            module_name: Name of the test module (without test_ prefix)
            
        Returns:
            bool: True if tests passed, False otherwise
        """
        test_file = self.tests_dir / f"test_{module_name}.py"
        
        if not test_file.exists():
            self.print_warning(f"Test file not found: {test_file}")
            return False
        
        self.print_step(f"Running {module_name} tests...")
        
        # Prepare pytest arguments
        pytest_args = [
            str(test_file),
            "--tb=short",  # Short traceback format
        ]
        
        if self.verbose:
            pytest_args.extend(["-v", "-s"])
        
        # Run pytest
        try:
            result = pytest.main(pytest_args)
            if result == 0:
                self.print_success(f"{module_name} tests passed")
                return True
            else:
                self.print_error(f"{module_name} tests failed")
                return False
        except Exception as e:
            self.print_error(f"Error running {module_name} tests: {str(e)}")
            return False
    
    def run_programmatic_tests(self, module_name):
        """
        Run tests programmatically (fallback if pytest not available).
        
        Args:
            module_name: Name of the test module
            
        Returns:
            bool: True if tests passed, False otherwise
        """
        try:
            if module_name == "data_processor":
                from tests.test_data_processor import run_dataprocessor_tests
                return run_dataprocessor_tests()
            elif module_name == "feature_engineer":
                from tests.test_feature_engineer import run_feature_engineer_tests
                return run_feature_engineer_tests()
            elif module_name == "model_trainer":
                from tests.test_model_trainer import run_model_trainer_tests
                return run_model_trainer_tests()
            elif module_name == "evaluator":
                from tests.test_evaluator import run_evaluator_tests
                return run_evaluator_tests()
            else:
                self.print_warning(f"No programmatic runner for {module_name}")
                return False
        except Exception as e:
            self.print_error(f"Error running {module_name} tests: {str(e)}")
            return False
    
    def run_single_module(self, module_name):
        """
        Run tests for a single module.
        
        Args:
            module_name: Name of the module to test
            
        Returns:
            bool: True if tests passed, False otherwise
        """
        if PYTEST_AVAILABLE:
            return self.run_pytest_module(module_name)
        else:
            self.print_warning("pytest not available, using programmatic runner")
            return self.run_programmatic_tests(module_name)
    
    def get_available_modules(self):
        """Get list of available test modules."""
        available_modules = []
        
        # Check for test files
        if self.tests_dir.exists():
            for test_file in self.tests_dir.glob("test_*.py"):
                module_name = test_file.stem[5:]  # Remove "test_" prefix
                available_modules.append(module_name)
        
        return available_modules
    
    def run_all_tests(self):
        """
        Run all available tests.
        
        Returns:
            bool: True if all tests passed, False otherwise
        """
        available_modules = self.get_available_modules()
        
        if not available_modules:
            self.print_warning("No test modules found")
            return False
        
        self.print_step(f"Found {len(available_modules)} test modules: {', '.join(available_modules)}")
        
        all_passed = True
        for module in available_modules:
            passed = self.run_single_module(module)
            self.results[module] = passed
            if not passed:
                all_passed = False
        
        return all_passed
    
    def run_with_coverage(self, module_name=None):
        """
        Run tests with coverage reporting.
        
        Args:
            module_name: Specific module to test (None for all)
            
        Returns:
            bool: True if tests passed, False otherwise
        """
        if not COVERAGE_AVAILABLE:
            self.print_error("Coverage not available. Install with: uv add --dev coverage")
            return False
        
        self.print_step("Running tests with coverage...")
        
        # Prepare coverage command
        cmd = ["coverage", "run", "-m", "pytest"]
        
        if module_name:
            cmd.append(f"tests/test_{module_name}.py")
        else:
            cmd.append("tests/")
        
        if self.verbose:
            cmd.extend(["-v", "-s"])
        
        try:
            # Run tests with coverage
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.print_success("Tests passed with coverage")
                
                # Generate coverage report
                print("\n📊 Coverage Report:")
                subprocess.run(["coverage", "report"], cwd=self.project_root)
                
                return True
            else:
                self.print_error("Tests failed")
                if self.verbose and result.stderr:
                    print(f"Error output: {result.stderr}")
                return False
                
        except Exception as e:
            self.print_error(f"Error running coverage: {str(e)}")
            return False
    
    def run_quick_validation(self):
        """
        Run quick validation tests to check basic functionality.
        
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.print_step("Running quick validation...")
        
        try:
            # Test basic imports
            from pipeline import DataProcessor
            from utils.config import TARGET_COL, CITY_COL
            
            # Test basic instantiation
            processor = DataProcessor()
            
            self.print_success("Quick validation passed")
            return True
            
        except Exception as e:
            self.print_error(f"Quick validation failed: {str(e)}")
            return False
    
    def export_detailed_results(self):
        """Export detailed test results in structured format."""
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Create detailed results structure
        detailed_results = {
            "execution_metadata": {
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": round(time.time() - self.start_time, 2) if self.start_time else 0,
                "runner_version": "Enhanced Test Runner v2.0",
                "environment": {
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    "working_directory": str(self.project_root)
                }
            },
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": round(success_rate, 2),
                "test_modules": list(self.results.keys()) if self.results else []
            },
            "detailed_results": self.results,
            "module_breakdown": {
                module: {
                    "passed": result,
                    "status": "PASS" if result else "FAIL"
                }
                for module, result in self.results.items()
            } if self.results else {}
        }
        
        return detailed_results
    
    def save_results_to_file(self, results):
        """Save test results to JSON file for instructor review."""
        try:
            with open(self.results_output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.print_success(f"Test results saved: {self.results_output_path}")
            
        except Exception as e:
            self.print_warning(f"Failed to save results file: {e}")
    
    def generate_cryptographic_proof(self, results):
        """Generate cryptographic proof of test execution."""
        if not self.generate_proof or not create_test_proof:
            return True
        
        self.print_step("Generating cryptographic proof...")
        
        try:
            # Create the proof using our crypto utilities
            proof = create_test_proof(
                test_results=results,
                repo_path=self.project_root,
                public_key_path=self.public_key_path
            )
            
            # Save proof to file
            with open(self.proof_output_path, 'w') as f:
                json.dump(proof, f, indent=2, default=str)
            
            self.print_success(f"Cryptographic proof generated: {self.proof_output_path}")
            
            # Show key proof information
            if self.verbose:
                git_info = proof.get('git_info', {})
                self.print_step(f"Proof commit: {git_info.get('commit_hash', 'unknown')[:8]}")
                self.print_step(f"Proof author: {git_info.get('author_name', 'unknown')}")
            
            return True
            
        except Exception as e:
            if 'CryptoError' in str(type(e)):
                self.print_error(f"Cryptographic proof generation failed: {e}")
            else:
                self.print_error(f"Unexpected error during proof generation: {e}")
            return False
    
    def print_summary(self):
        """Print test execution summary."""
        if not self.results:
            return
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n📋 TEST SUMMARY")
        print(f"{'='*40}")
        
        passed = sum(1 for result in self.results.values() if result)
        total = len(self.results)
        
        for module, result in self.results.items():
            status = "PASSED" if result else "FAILED"
            icon = "✅" if result else "❌"
            print(f"{icon} {module:<20} {status}")
        
        print(f"{'='*40}")
        print(f"Total: {passed}/{total} passed")
        if elapsed > 0:
            print(f"Time: {elapsed:.1f}s")
        
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print("❌ Some tests failed!")
        
        # Add crypto summary
        if self.generate_proof and CRYPTO_AVAILABLE:
            print(f"\n🔐 CRYPTOGRAPHIC VERIFICATION")
            print(f"{'='*40}")
            
            if self.proof_output_path.exists():
                print(f"✅ Proof file generated: {self.proof_output_path.name}")
                print(f"📋 Results file: {self.results_output_path.name}")
                print(f"🔑 Public key used: {self.public_key_path.name}")
                print(f"\n📝 Instructions:")
                print(f"1. Commit and push these files to your repository")
                print(f"2. Provide your git repository URL to instructor")
                print(f"3. Instructor will verify your proof cryptographically")
            else:
                print(f"❌ Proof generation failed")
                print(f"📋 Results saved but cannot be verified")
        
        elif not CRYPTO_AVAILABLE:
            print(f"\n⚠️  Cryptographic verification not available")
            if CRYPTOGRAPHY_LIB_AVAILABLE:
                print(f"Missing crypto_utils.py in scripts/ directory")
            else:
                print(f"Install with: uv add cryptography")
        
        else:
            print(f"\n🔓 Proof generation disabled")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run technical validation tests for Air Quality ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py                    # Run all tests with proof
  python scripts/run_tests.py --verbose          # Verbose output
  python scripts/run_tests.py -m data_processor  # Specific module
  python scripts/run_tests.py --coverage         # With coverage report
  python scripts/run_tests.py --quick            # Quick validation
  python scripts/run_tests.py --no-proof         # Skip proof generation
        """
    )
    
    parser.add_argument(
        "-m", "--module",
        help="Run tests for specific module (e.g., data_processor)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage reporting (no proof generation)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation only"
    )
    parser.add_argument(
        "--no-proof",
        action="store_true",
        help="Skip cryptographic proof generation"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    generate_proof = not args.no_proof and not args.coverage  # No proof with coverage
    runner = TestRunner(verbose=args.verbose, generate_proof=generate_proof)
    runner.start_time = time.time()
    
    # Print header
    runner.print_header("Air Quality ML Pipeline - Test Runner with Crypto Verification")
    
    # Check setup
    if not runner.check_dependencies() or not runner.check_project_structure():
        sys.exit(1)
    
    if not runner.check_crypto_setup():
        if runner.generate_proof:
            print("🔄 Continuing without proof generation...")
            runner.generate_proof = False
    
    if not runner.check_git_status():
        sys.exit(1)
    
    # Run tests based on arguments
    success = False
    
    try:
        if args.quick:
            success = runner.run_quick_validation()
        elif args.coverage:
            success = runner.run_with_coverage(args.module)
        elif args.module:
            success = runner.run_single_module(args.module)
            runner.results[args.module] = success
        else:
            success = runner.run_all_tests()
        
        # Generate detailed results and proof (if not quick mode and tests were run)
        if not args.quick and not args.coverage and runner.results:
            detailed_results = runner.export_detailed_results()
            runner.save_results_to_file(detailed_results)

            if runner.generate_proof:
                proof_success = runner.generate_cryptographic_proof(detailed_results)
                if not proof_success:
                    runner.print_warning("Proof generation failed")
        
        # Print summary
        runner.print_summary()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
