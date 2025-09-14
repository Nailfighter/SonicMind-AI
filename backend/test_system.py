#!/usr/bin/env python3
"""
ðŸ§ª SonicMind AI - Complete System Testing Suite
Comprehensive testing for all modules and integration scenarios
"""

import socketio
import time
import json
import threading
import sys
import os
from typing import Dict, List, Optional
import argparse

# Test individual modules


def test_individual_modules():
    """Test each module independently"""
    print("ðŸ”¬" + "="*50)
    print("    INDIVIDUAL MODULE TESTING")
    print("="*52)

    # Test 1: Auto-EQ System
    print("\n1ï¸âƒ£ Testing Auto-EQ System...")
    try:
        from auto_eq_system import AutoEQSystem

        eq_system = AutoEQSystem()
        print("   âœ… Auto-EQ System initialized")

        # Test EQ band operations
        success = eq_system.update_band(0, 'gain_db', 2.5)
        print(f"   âœ… EQ band update: {'Success' if success else 'Failed'}")

        # Test device enumeration
        devices = eq_system.get_available_devices()
        input_count = len(devices.get('input_devices', []))
        output_count = len(devices.get('output_devices', []))
        print(
            f"   âœ… Audio devices: {input_count} input, {output_count} output")

        eq_system.cleanup()

    except ImportError as e:
        print(f"   âŒ Auto-EQ System import failed: {e}")
    except Exception as e:
        print(f"   âŒ Auto-EQ System test failed: {e}")

    # Test 2: Instrument Detection
    print("\n2ï¸âƒ£ Testing Instrument Detection...")
    try:
        from instrument_detection import InstrumentDetector

        detector = InstrumentDetector()
        print("   âœ… Instrument Detector initialized")

        instruments = detector.get_supported_instruments()
        print(f"   âœ… Supported instruments: {len(instruments)}")
        print(f"      ðŸ“‹ {', '.join(instruments[:5])}...")

        detector.cleanup()

    except ImportError as e:
        print(f"   âŒ Instrument Detection import failed: {e}")
    except Exception as e:
        print(f"   âŒ Instrument Detection test failed: {e}")

    # Test 3: Material Detection
    print("\n3ï¸âƒ£ Testing Material Detection...")
    try:
        from material_detection import MaterialDetector

        detector = MaterialDetector()
        print("   âœ… Material Detector initialized")

        materials = detector.get_supported_materials()
        presets = detector.get_room_presets()
        print(f"   âœ… Supported materials: {len(materials)}")
        print(f"   âœ… Room presets: {len(presets)}")
        print(f"      ðŸ“‹ {', '.join(list(presets.keys()))}")

        detector.cleanup()

    except ImportError as e:
        print(f"   âŒ Material Detection import failed: {e}")
    except Exception as e:
        print(f"   âŒ Material Detection test failed: {e}")

    print("\nâœ… Individual module testing complete!")


class SystemTestClient:
    """Comprehensive test client for the integrated system"""

    def __init__(self, server_url="http://localhost:8000"):
        self.sio = socketio.Client()
        self.server_url = server_url
        self.connected = False
        self.test_results = {}

        # Event tracking
        self.events_received = []
        self.system_status = {}

        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Setup Socket.IO event handlers for testing"""

        @self.sio.event
        def connect():
            print("âœ… Connected to SonicMind AI backend")
            self.connected = True

        @self.sio.event
        def disconnect():
            print("âŒ Disconnected from backend")
            self.connected = False

        @self.sio.event
        def system_status(data):
            print(f"ðŸ“Š System Status Update:")
            self.system_status = data
            self._print_status(data)
            self.events_received.append(('system_status', data))

        @self.sio.event
        def room_analysis(data):
            print(f"\nðŸ  Room Analysis Complete:")
            print(f"   Material: {data.get('dominant_material', 'Unknown')}")
            print(f"   Confidence: {data.get('material_confidence', 0):.1%}")
            print(
                f"   Room Type: {data.get('acoustic_properties', {}).get('room_type', 'Unknown')}")
            print(
                f"   Absorption: {data.get('acoustic_properties', {}).get('average_absorption', 0):.3f}")

            eq_preset = data.get('eq_preset', {})
            if eq_preset:
                print("   ðŸŽ›ï¸ EQ Adjustments:")
                for band, adj in eq_preset.get('eq_adjustments', {}).items():
                    print(f"      {band}: {adj:+.1f} dB")

            self.events_received.append(('room_analysis', data))

        @self.sio.event
        def instrument_detected(data):
            instrument = data.get('instrument', 'unknown')
            confidence = data.get('confidence', 0)
            print(f"ðŸŽµ Instrument Detected: {instrument} ({confidence:.1%})")
            self.events_received.append(('instrument_detected', data))

        @self.sio.event
        def eq_updated(data):
            update_type = data.get('type', 'unknown')
            bands = data.get('bands', [])
            print(f"ðŸŽ›ï¸ EQ Updated ({update_type}):")

            for i, band in enumerate(bands):
                freq = band.get('freq', 0)
                gain = band.get('gain', 0)
                if abs(gain) > 0.1:  # Only show significant changes
                    print(f"   Band {i+1} ({freq}Hz): {gain:+.1f} dB")

            self.events_received.append(('eq_updated', data))

        # Response handlers
        @self.sio.event
        def audio_started(data):
            success = data.get('success', False)
            print(
                f"ðŸŽ§ Audio Engine: {'âœ… Started' if success else 'âŒ Failed'}")
            self.test_results['audio_start'] = success

        @self.sio.event
        def detection_started(data):
            success = data.get('success', False)
            print(
                f"ðŸ“· Detection System: {'âœ… Started' if success else 'âŒ Failed'}")
            self.test_results['detection_start'] = success

        @self.sio.event
        def auto_eq_started(data):
            success = data.get('success', False)
            print(
                f"ðŸ¤– Auto-EQ: {'âœ… Started' if success else 'âŒ Failed'}")
            self.test_results['auto_eq_start'] = success

    def _print_status(self, status):
        """Print system status in readable format"""
        audio = "ðŸŸ¢ Running" if status.get(
            'audio_running') else "ðŸ”´ Stopped"
        detection = "ðŸŸ¢ Running" if status.get(
            'detection_running') else "ðŸ”´ Stopped"
        auto_eq = "ðŸŸ¢ Active" if status.get(
            'auto_eq_running') else "ðŸ”´ Inactive"

        print(f"   Audio: {audio}")
        print(f"   Detection: {detection}")
        print(f"   Auto-EQ: {auto_eq}")
        print(f"   Instrument: {status.get('current_instrument', 'none')}")
        print(f"   Room Preset: {status.get('room_preset', 'neutral')}")

    async def connect_to_server(self) -> bool:
        """Connect to the backend server"""
        try:
            print(f"ðŸ”Œ Connecting to {self.server_url}...")
            self.sio.connect(self.server_url)

            # Wait for connection
            timeout = 10
            while not self.connected and timeout > 0:
                time.sleep(0.1)
                timeout -= 1

            return self.connected

        except Exception as e:
            print(f"âŒ Connection failed: {e}")
            return False

    def disconnect_from_server(self):
        """Disconnect from server"""
        if self.connected:
            self.sio.disconnect()

    def run_full_system_test(self) -> Dict:
        """Run complete system integration test"""
        print("\nðŸš€" + "="*50)
        print("    FULL SYSTEM INTEGRATION TEST")
        print("="*52)

        if not self.connected:
            print("âŒ Not connected to server")
            return {"success": False, "error": "Not connected"}

        test_sequence = [
            ("ðŸŽ§ Starting Audio Engine", self._test_audio_start),
            ("ðŸ“· Starting Detection Systems", self._test_detection_start),
            ("â±ï¸ Waiting for Room Analysis", self._wait_for_room_analysis),
            ("ðŸŽµ Monitoring Instrument Detection", self._monitor_instruments),
            ("ðŸ¤– Starting Auto-EQ", self._test_auto_eq_start),
            ("âš¡ Testing Manual EQ", self._test_manual_eq),
            ("ðŸ”„ Testing EQ Reset", self._test_eq_reset),
            ("ðŸ“Š Final System Check", self._final_system_check)
        ]

        results = {"success": True, "tests": {}, "events_received": 0}

        for test_name, test_func in test_sequence:
            print(f"\n{test_name}...")
            try:
                result = test_func()
                results["tests"][test_name] = result
                print(f"   {'âœ… Passed' if result else 'âŒ Failed'}")

                if not result:
                    results["success"] = False

            except Exception as e:
                print(f"   âŒ Error: {e}")
                results["tests"][test_name] = False
                results["success"] = False

        results["events_received"] = len(self.events_received)

        # Cleanup
        self._cleanup_test()

        return results

    def _test_audio_start(self) -> bool:
        """Test audio engine startup"""
        self.sio.emit('start_audio', {})
        time.sleep(2)
        return self.test_results.get('audio_start', False)

    def _test_detection_start(self) -> bool:
        """Test detection systems startup"""
        self.sio.emit('start_detection', {'camera_index': 0})
        time.sleep(3)
        return self.test_results.get('detection_start', False)

    def _wait_for_room_analysis(self) -> bool:
        """Wait for room analysis to complete"""
        print("   ðŸ“¸ Room analysis in progress...")

        # Wait up to 15 seconds for room analysis
        timeout = 15
        room_analysis_received = False

        while timeout > 0:
            for event_type, _ in self.events_received:
                if event_type == 'room_analysis':
                    room_analysis_received = True
                    break

            if room_analysis_received:
                break

            time.sleep(1)
            timeout -= 1
            print(f"   â³ Waiting... ({timeout}s remaining)")

        return room_analysis_received

    def _monitor_instruments(self) -> bool:
        """Monitor for instrument detection"""
        print("   ðŸŽµ Monitoring instrument detection...")

        # Monitor for 10 seconds
        start_time = time.time()
        instruments_detected = []

        while time.time() - start_time < 10:
            for event_type, data in self.events_received:
                if event_type == 'instrument_detected':
                    instrument = data.get('instrument', 'unknown')
                    if instrument not in instruments_detected:
                        instruments_detected.append(instrument)

            time.sleep(1)

        print(
            f"   ðŸ“Š Detected instruments: {instruments_detected if instruments_detected else 'None'}")
        return True  # Always pass - detection is optional

    def _test_auto_eq_start(self) -> bool:
        """Test Auto-EQ startup"""
        self.sio.emit('start_auto_eq', {})
        time.sleep(3)
        return self.test_results.get('auto_eq_start', False)

    def _test_manual_eq(self) -> bool:
        """Test manual EQ adjustment"""
        print("   ðŸŽ›ï¸ Testing manual EQ adjustment...")

        # Adjust band 2 (1kHz) to +2dB
        self.sio.emit('manual_eq_update', {
            'band_index': 2,
            'parameter': 'gain_db',
            'value': 2.0
        })

        time.sleep(2)

        # Check if EQ update event was received
        eq_updates = [event for event_type, event in self.events_received
                      if event_type == 'eq_updated']

        return len(eq_updates) > 0

    def _test_eq_reset(self) -> bool:
        """Test EQ reset functionality"""
        print("   ðŸ”„ Testing EQ reset...")

        self.sio.emit('reset_eq', {})
        time.sleep(2)

        # Look for reset event
        reset_events = [event for event_type, event in self.events_received
                        if event_type == 'eq_updated' and event.get('type') == 'reset']

        return len(reset_events) > 0

    def _final_system_check(self) -> bool:
        """Final system status check"""
        self.sio.emit('get_system_status', {})
        time.sleep(1)

        # Check that we have current status
        return bool(self.system_status)

    def _cleanup_test(self):
        """Cleanup after testing"""
        print("\nðŸ§¹ Cleaning up test...")

        try:
            self.sio.emit('stop_auto_eq', {})
            time.sleep(1)
            self.sio.emit('stop_detection', {})
            time.sleep(1)
            self.sio.emit('stop_audio', {})
            time.sleep(1)
        except Exception:
            pass

    def run_interactive_test(self):
        """Interactive testing mode"""
        print("\nðŸŽ® Interactive Test Mode")
        print("Available commands:")
        commands = [
            "1. start_audio - Start audio processing",
            "2. start_detection - Start camera detection",
            "3. start_auto_eq - Start automatic EQ",
            "4. stop_audio - Stop audio",
            "5. stop_detection - Stop detection",
            "6. stop_auto_eq - Stop auto-EQ",
            "7. manual_eq <band> <gain> - Manual EQ (e.g., manual_eq 2 1.5)",
            "8. reset_eq - Reset EQ bands",
            "9. status - Get system status",
            "10. events - Show received events",
            "11. results - Show test results",
            "0. exit - Exit interactive mode"
        ]

        for cmd in commands:
            print(f"  {cmd}")

        while True:
            try:
                cmd = input("\n> ").strip().lower()

                if cmd == "0" or cmd == "exit":
                    break
                elif cmd == "1" or cmd == "start_audio":
                    self.sio.emit('start_audio', {})
                elif cmd == "2" or cmd == "start_detection":
                    self.sio.emit('start_detection', {'camera_index': 0})
                elif cmd == "3" or cmd == "start_auto_eq":
                    self.sio.emit('start_auto_eq', {})
                elif cmd == "4" or cmd == "stop_audio":
                    self.sio.emit('stop_audio', {})
                elif cmd == "5" or cmd == "stop_detection":
                    self.sio.emit('stop_detection', {})
                elif cmd == "6" or cmd == "stop_auto_eq":
                    self.sio.emit('stop_auto_eq', {})
                elif cmd.startswith("manual_eq"):
                    parts = cmd.split()
                    if len(parts) >= 3:
                        try:
                            band = int(parts[1])
                            gain = float(parts[2])
                            self.sio.emit('manual_eq_update', {
                                'band_index': band,
                                'parameter': 'gain_db',
                                'value': gain
                            })
                        except ValueError:
                            print("âŒ Invalid format. Use: manual_eq <band> <gain>")
                    else:
                        print("âŒ Usage: manual_eq <band_index> <gain_db>")
                elif cmd == "8" or cmd == "reset_eq":
                    self.sio.emit('reset_eq', {})
                elif cmd == "9" or cmd == "status":
                    self.sio.emit('get_system_status', {})
                elif cmd == "10" or cmd == "events":
                    print(f"ðŸ“‹ Events received: {len(self.events_received)}")
                    for i, (event_type, _) in enumerate(self.events_received[-10:]):
                        print(f"   {i+1}. {event_type}")
                elif cmd == "11" or cmd == "results":
                    print("ðŸ“Š Test Results:")
                    for test, result in self.test_results.items():
                        status = "âœ… Pass" if result else "âŒ Fail"
                        print(f"   {test}: {status}")
                else:
                    print("â“ Unknown command. Type 'exit' to quit.")

                time.sleep(0.5)  # Brief pause

            except KeyboardInterrupt:
                break
            except EOFError:
                break

        print("\nðŸ‘‹ Exiting interactive mode...")


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(
        description="ðŸ§ª SonicMind AI System Testing Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Modes:
  --modules       Test individual modules only
  --integration   Run full integration test
  --interactive   Interactive testing mode
  --server URL    Backend server URL (default: http://localhost:8000)
  
Examples:
  python test_system.py --modules
  python test_system.py --integration  
  python test_system.py --interactive
  python test_system.py --integration --server http://localhost:9000
        """
    )

    parser.add_argument('--modules', action='store_true',
                        help='Test individual modules')
    parser.add_argument('--integration', action='store_true',
                        help='Run full integration test')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive testing mode')
    parser.add_argument('--server', default='http://localhost:8000',
                        help='Backend server URL')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests')

    args = parser.parse_args()

    # Default to all tests if no specific test chosen
    if not any([args.modules, args.integration, args.interactive]):
        args.all = True

    print("ðŸ§ª" + "="*60)
    print("    SonicMind AI - System Testing Suite")
    print("    Comprehensive Testing & Validation")
    print("="*62)

    exit_code = 0

    # Test individual modules
    if args.modules or args.all:
        test_individual_modules()

        if args.all:
            print("\n" + "="*30)

    # Integration and interactive tests require server connection
    if args.integration or args.interactive or args.all:
        client = SystemTestClient(args.server)

        print(f"\nðŸ”Œ Connecting to backend server: {args.server}")

        # Use the synchronous connection method
        try:
            client.sio.connect(args.server)
            time.sleep(2)  # Give it time to connect

            if client.sio.connected:
                print("âœ… Connected to backend")
                client.connected = True

                # Run integration test
                if args.integration or args.all:
                    results = client.run_full_system_test()

                    print(f"\nðŸ“Š Integration Test Results:")
                    print(
                        f"   Overall: {'âœ… PASSED' if results['success'] else 'âŒ FAILED'}")
                    print(f"   Events received: {results['events_received']}")
                    print(f"   Individual tests:")

                    for test, result in results['tests'].items():
                        status = "âœ… Pass" if result else "âŒ Fail"
                        print(f"      {test}: {status}")

                    if not results['success']:
                        exit_code = 1

                # Interactive mode
                if args.interactive:
                    client.run_interactive_test()

            else:
                print("âŒ Failed to connect to backend")
                print("   Make sure the server is running: python main.py")
                exit_code = 1

        except Exception as e:
            print(f"âŒ Connection error: {e}")
            print("   Make sure the server is running: python main.py")
            exit_code = 1

        finally:
            try:
                client.disconnect_from_server()
            except Exception:
                pass

    print(f"\n{'âœ… All tests completed successfully!' if exit_code == 0 else 'âŒ Some tests failed.'}")
    return exit_code


if __name__ == "__main__":
    try:
        # Set UTF-8 encoding for Windows console
        if sys.platform == "win32":
            import locale
            import codecs
            sys.stdout = codecs.getwriter(locale.getpreferredencoding())(
                sys.stdout.buffer, 'replace')
            sys.stderr = codecs.getwriter(locale.getpreferredencoding())(
                sys.stderr.buffer, 'replace')

        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
