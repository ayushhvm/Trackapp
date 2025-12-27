#!/usr/bin/env python3
"""
Raspberry Pi Camera Attendance Capture Script

Captures a photo using the Raspberry Pi camera module and sends it to the Django API
for face recognition and attendance marking.

Requirements:
    - picamera2 library (for Raspberry Pi Camera Module)
    - requests library
    - config.json file with API settings

Usage:
    python capture_and_send.py

Configuration:
    Edit config.json to set API URL, session ID, location, etc.
"""

import json
import os
import sys
import requests
from datetime import datetime
from pathlib import Path

# Try to import picamera2, but handle gracefully if not on Pi
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available. Running in test mode (will use existing image if available).")


def load_config():
    """Load configuration from config.json file"""
    script_dir = Path(__file__).parent
    config_path = script_dir / 'config.json'
    
    if not config_path.exists():
        print(f"Error: config.json not found at {config_path}")
        print("Please create config.json with API settings.")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['api_url', 'session_id']
    for field in required_fields:
        if field not in config:
            print(f"Error: Required field '{field}' not found in config.json")
            sys.exit(1)
    
    return config


def capture_photo(output_path='captured_photo.jpg', width=1920, height=1080):
    """
    Capture a photo using the Raspberry Pi camera module.
    
    Args:
        output_path: Path to save the captured photo
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Path to the captured image file, or None if capture failed
    """
    if not PICAMERA2_AVAILABLE:
        # Test mode: check if output_path exists
        if os.path.exists(output_path):
            print(f"Test mode: Using existing image at {output_path}")
            return output_path
        else:
            print(f"Error: picamera2 not available and test image not found at {output_path}")
            return None
    
    try:
        # Initialize camera
        camera = Picamera2()
        
        # Configure camera
        config = camera.create_still_configuration(main={"size": (width, height)})
        camera.configure(config)
        
        # Start camera
        camera.start()
        
        # Wait for camera to stabilize
        import time
        time.sleep(2)
        
        # Capture photo
        camera.capture_file(output_path)
        
        # Stop camera
        camera.stop()
        camera.close()
        
        print(f"Photo captured successfully: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error capturing photo: {str(e)}")
        return None


def send_to_api(image_path, config):
    """
    Send the captured image to the Django API for processing.
    
    Args:
        image_path: Path to the image file
        config: Configuration dictionary
    
    Returns:
        API response as dictionary, or None if request failed
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    # Prepare the data
    api_url = config['api_url']
    session_id = config['session_id']
    
    # Get current timestamp (when photo was captured)
    captured_at = datetime.now().isoformat()
    
    # Prepare multipart form data
    with open(image_path, 'rb') as image_file:
        files = {
            'image': (os.path.basename(image_path), image_file, 'image/jpeg')
        }
        
        data = {
            'session_id': str(session_id),
            'captured_at': captured_at,
            'threshold': str(config.get('threshold', 0.6)),
        }
        
        # Add optional fields
        if 'location_name' in config:
            data['location_name'] = config['location_name']
        if 'device_id' in config:
            data['device_id'] = config['device_id']
        if 'latitude' in config:
            data['latitude'] = str(config['latitude'])
        if 'longitude' in config:
            data['longitude'] = str(config['longitude'])
        
        try:
            print(f"Sending image to API: {api_url}")
            response = requests.post(api_url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"API request failed with status {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error message: {error_data.get('message', 'Unknown error')}")
                except:
                    print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Network error: {str(e)}")
            return None


def display_results(result):
    """Display the API response results in a readable format"""
    if not result:
        return
    
    print("\n" + "="*50)
    print("ATTENDANCE MARKING RESULTS")
    print("="*50)
    
    if result.get('success'):
        print(f"Status: ✓ Success")
        print(f"Message: {result.get('message', 'N/A')}")
        print(f"Students Marked: {result.get('marked_count', 0)}")
        
        recognitions = result.get('recognitions', [])
        if recognitions:
            print("\nRecognition Details:")
            print("-" * 50)
            for rec in recognitions:
                student_id = rec.get('student_id', 'Unknown')
                name = rec.get('name', 'Unknown')
                confidence = rec.get('confidence', 0.0)
                status = rec.get('status', 'unknown')
                
                status_symbol = {
                    'marked': '✓',
                    'already_marked': '○',
                    'not_found': '✗',
                    'unrecognized': '?'
                }.get(status, '?')
                
                print(f"{status_symbol} {student_id} - {name}")
                print(f"  Confidence: {confidence:.2%} | Status: {status}")
        else:
            print("\nNo students recognized.")
    else:
        print(f"Status: ✗ Failed")
        print(f"Message: {result.get('message', 'Unknown error')}")
    
    print("="*50 + "\n")


def main():
    """Main function to capture photo and send to API"""
    print("="*50)
    print("Raspberry Pi Attendance Capture")
    print("="*50)
    
    # Load configuration
    config = load_config()
    print(f"Configuration loaded from config.json")
    print(f"API URL: {config['api_url']}")
    print(f"Session ID: {config['session_id']}")
    if 'location_name' in config:
        print(f"Location: {config['location_name']}")
    if 'device_id' in config:
        print(f"Device ID: {config['device_id']}")
    print()
    
    # Capture photo
    output_path = config.get('output_path', 'captured_photo.jpg')
    image_path = capture_photo(output_path)
    
    if not image_path:
        print("Failed to capture photo. Exiting.")
        sys.exit(1)
    
    # Send to API
    result = send_to_api(image_path, config)
    
    # Display results
    display_results(result)
    
    # Save local copy if configured
    if config.get('save_local_copy', True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_path = f"captured_photo_{timestamp}.jpg"
        import shutil
        shutil.copy2(image_path, saved_path)
        print(f"Photo saved locally: {saved_path}")
    
    # Clean up temporary file if configured
    if not config.get('save_local_copy', True) and image_path == output_path:
        try:
            os.remove(image_path)
        except:
            pass
    
    # Exit with appropriate code
    if result and result.get('success'):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

