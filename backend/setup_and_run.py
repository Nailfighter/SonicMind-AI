#!/usr/bin/env python3
"""
ðŸš€ SonicMind AI - Easy Setup & Launch Script
One-command setup and testing for the modular Auto-EQ system
"""

import os
import sys
import subprocess
import argparse
import time


def check_file_exists(filename):
    """Check if required file exists"""
    exists = os.path.exists(filename)
    status = "âœ…" if exists else "âŒ"
    print(f"   {status} {filename}")
    return exists


def check_system_files():
    """Check if all system files are present"""
    print("ðŸ“ Checking system files...")

    required_files = [
        'main.py',
        'auto_eq_system.py',
        'instrument_detection.py',
        'material_detection.py',
        'test_system.py',
        'requirements.txt'
    ]

    all_present = True
    for filename in required_files:
        if not check_file_exists(filename):
            all_present = False

    return all_present


def install_dependencies():
    """Install required dependencies"""
    print("\nðŸ“¦ Installing dependencies...")

    try:
        # Install basic requirements
        print("   Installing basic packages...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True)
        print("   âœ… Basic packages installed")

        # Try to install CLIP
        print("   Installing CLIP...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "git+https://github.com/openai/CLIP.git"
            ], check=True, capture_output=True)
            print("   âœ… Official CLIP installed")
        except subprocess.CalledProcessError:
            print("   âš ï¸ Official CLIP failed, trying alternative...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "open-clip-torch"
                ], check=True, capture_output=True)
                print("   âœ… Alternative CLIP installed")
            except subprocess.CalledProcessError:
                print(
                    "   âš ï¸ CLIP installation failed - detection features will use mock data")

        return True

    except subprocess.CalledProcessError as e:
        print(f"   âŒ Installation failed: {e}")
        return False


def test_imports():
    """Test that all imports work"""
    print("\nðŸ§ª Testing imports...")

    import_tests = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('sounddevice', 'SoundDevice'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('socketio', 'Socket.IO'),
        ('flask', 'Flask'),
        ('torch', 'PyTorch'),
    ]

    failed_imports = []

    for module, name in import_tests:
        try:
            __import__(module)
            print(f"   âœ… {name}")
        except ImportError:
            print(f"   âŒ {name}")
            failed_imports.append(name)

    # Test CLIP separately
    try:
        import clip
        print("   âœ… CLIP (Official)")
    except ImportError:
        try:
            import open_clip
            print("   âœ… CLIP (Alternative)")
        except ImportError:
            print("   âš ï¸ CLIP (Will use mock data)")

    return len(failed_imports) == 0


def run_module_tests():
    """Run individual module tests"""
    print("\nðŸ”¬ Running module tests...")

    try:
        result = subprocess.run([
            sys.executable, "test_system.py", "--modules"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("   âœ… Module tests passed")
            return True
        else:
            print("   âŒ Module tests failed")
            print(f"   Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("   âš ï¸ Module tests timed out")
        return False
    except Exception as e:
        print(f"   âŒ Module test error: {e}")
        return False


def start_server(background=False):
    """Start the main server"""
    print(f"\nðŸš€ Starting SonicMind AI server...")

    if background:
        # Start in background
        process = subprocess.Popen([sys.executable, "main.py"])
        print(f"   ðŸŒ Server starting in background (PID: {process.pid})")
        print("   ðŸ”Œ Socket.IO endpoint: ws://localhost:8000/socket.io/")
        return process
    else:
        # Start in foreground
        print("   ðŸŒ Server starting at: http://localhost:8000")
        print("   ðŸ”Œ Socket.IO endpoint: ws://localhost:8000/socket.io/")
        print("   ðŸ›‘ Press Ctrl+C to stop")
        print("-" * 50)

        try:
            subprocess.run([sys.executable, "main.py"])
        except KeyboardInterrupt:
            print("\nðŸ›‘ Server stopped!")


def run_integration_tests():
    """Run integration tests against running server"""
    print("\nðŸ§ª Running integration tests...")

    # Give server time to start
    time.sleep(3)

    try:
        result = subprocess.run([
            sys.executable, "test_system.py", "--integration"
        ], timeout=120)  # 2 minute timeout

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("   âš ï¸ Integration tests timed out")
        return False
    except Exception as e:
        print(f"   âŒ Integration test error: {e}")
        return False


def interactive_testing():
    """Run interactive testing mode"""
    print("\nðŸŽ® Starting interactive testing...")

    try:
        subprocess.run([sys.executable, "test_system.py", "--interactive"])
    except KeyboardInterrupt:
        print("\nðŸ‘‹ Interactive testing stopped!")


def main():
    parser = argparse.ArgumentParser(
        description="ðŸš€ SonicMind AI Setup & Launch Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python setup_and_run.py                    # Full setup + start server
  python setup_and_run.py --install-only     # Just install dependencies  
  python setup_and_run.py --test-only        # Just run tests
  python setup_and_run.py --server-only      # Just start server
  python setup_and_run.py --interactive      # Interactive testing
  python setup_and_run.py --full-test        # Complete test suite
        """
    )

    parser.add_argument('--install-only', action='store_true',
                        help='Only install dependencies')
    parser.add_argument('--test-only', action='store_true',
                        help='Only run tests (no server)')
    parser.add_argument('--server-only', action='store_true',
                        help='Only start server (skip setup)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive testing mode')
    parser.add_argument('--full-test', action='store_true',
                        help='Run complete test suite')
    parser.add_argument('--skip-install', action='store_true',
                        help='Skip dependency installation')

    args = parser.parse_args()

    print("ðŸš€" + "="*60)
    print("    SonicMind AI - Easy Setup & Launch")
    print("    Modular Auto-EQ System")
    print("="*62)

    # Check system files first
    if not check_system_files():
        print("\nâŒ Missing required files! Make sure all modules are present.")
        return 1

    success = True

    # Install dependencies (unless skipped)
    if not args.skip_install and not args.server_only:
        if not install_dependencies():
            print("\nâš ï¸ Dependency installation had issues, but continuing...")

    # Test imports
    if not args.server_only:
        if not test_imports():
            print("\nâš ï¸ Some imports failed, but system may still work...")

    # Install-only mode
    if args.install_only:
        print("\nâœ… Installation complete!")
        return 0

    # Test-only mode
    if args.test_only:
        success = run_module_tests()
        return 0 if success else 1

    # Server-only mode
    if args.server_only:
        start_server(background=False)
        return 0

    # Interactive testing mode
    if args.interactive:
        print("\nðŸŽ® Interactive Mode:")
        print("1. Starting server in background...")
        server_process = start_server(background=True)

        try:
            time.sleep(5)  # Give server time to start
            interactive_testing()
        finally:
            if server_process:
                print("ðŸ›‘ Stopping background server...")
                server_process.terminate()
                server_process.wait()

        return 0

    # Full test mode
    if args.full_test:
        print("\nðŸ§ª Running Full Test Suite...")

        # 1. Module tests
        print("\n1ï¸âƒ£ Module Tests:")
        module_success = run_module_tests()

        # 2. Start server in background
        print("\n2ï¸âƒ£ Starting Server:")
        server_process = start_server(background=True)

        # 3. Integration tests
        print("\n3ï¸âƒ£ Integration Tests:")
        integration_success = False

        try:
            integration_success = run_integration_tests()
        finally:
            if server_process:
                print("ðŸ›‘ Stopping test server...")
                server_process.terminate()
                server_process.wait()

        # Results
        print(f"\nðŸ“Š Test Results:")
        print(
            f"   Module Tests: {'âœ… Pass' if module_success else 'âŒ Fail'}")
        print(
            f"   Integration Tests: {'âœ… Pass' if integration_success else 'âŒ Fail'}")

        overall_success = module_success and integration_success
        print(
            f"   Overall: {'âœ… SUCCESS' if overall_success else 'âŒ FAILED'}")

        return 0 if overall_success else 1

    # Default mode: Quick setup + start server
    print("\nðŸŽ¯ Default Mode: Setup + Start Server")

    # Quick module test
    if not run_module_tests():
        print("âš ï¸ Module tests failed, but starting server anyway...")

    # Start server
    start_server(background=False)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nðŸ›‘ Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nâŒ Setup error: {e}")
        sys.exit(1)
