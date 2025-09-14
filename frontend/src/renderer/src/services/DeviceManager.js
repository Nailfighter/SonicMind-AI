import { useBackendAPI } from '../hooks/useBackendAPI'
import backendAPI from './BackendAPI'

/**
 * DeviceManager - Enhanced device management for SonicMind-AI
 * 
 * This service manages both web API devices (cameras/microphones) and 
 * backend audio devices, providing a unified interface for device 
 * selection and management.
 */
class DeviceManager {
  constructor() {
    this.webDevices = {
      cameras: [],
      microphones: [],
      speakers: []
    }
    
    this.backendAudioDevices = {
      input: [],
      output: []
    }
    
    this.selectedDevices = {
      // Web API devices (for cameras/microphones)
      audienceCamera: null,
      artistCamera: null,
      artistMicrophone: null,
      // Backend audio devices
      audioInput: null,
      audioOutput: null
    }
    
    this.deviceListeners = []
    this.isInitialized = false
    this.lastRefresh = 0
    this.refreshInterval = 5000 // 5 seconds
  }

  /**
   * Initialize device manager and discover all devices
   * @returns {Promise<boolean>} Success status
   */
  async initialize() {
    if (this.isInitialized) {
      return true
    }
    
    console.log('🎛️ Initializing DeviceManager')
    
    try {
      // Request permissions first (important for device labels)
      await this._requestPermissions()
      
      // Discover all devices
      await this.refreshDevices()
      
      this.isInitialized = true
      this._notifyListeners('initialized', { devices: this.getAllDevices() })
      
      return true
    } catch (error) {
      console.error('❌ DeviceManager initialization failed:', error)
      return false
    }
  }

  /**
   * Refresh all device lists
   * @param {boolean} force - Force refresh even if recently done
   * @returns {Promise<boolean>} Success status
   */
  async refreshDevices(force = false) {
    const now = Date.now()
    if (!force && (now - this.lastRefresh) < this.refreshInterval) {
      return true
    }
    
    console.log('🔄 Refreshing device lists')
    
    try {
      // Refresh web devices and backend audio devices in parallel
      const [webSuccess, backendSuccess] = await Promise.all([
        this._refreshWebDevices(),
        this._refreshBackendAudioDevices()
      ])
      
      this.lastRefresh = now
      this._notifyListeners('refreshed', { 
        web: webSuccess, 
        backend: backendSuccess,
        devices: this.getAllDevices()
      })
      
      return webSuccess && backendSuccess
    } catch (error) {
      console.error('❌ Device refresh failed:', error)
      return false
    }
  }

  /**
   * Get all devices in a unified format
   * @returns {Object} All available devices
   */
  getAllDevices() {
    return {
      // Web API devices
      cameras: this.webDevices.cameras,
      microphones: this.webDevices.microphones,
      speakers: this.webDevices.speakers,
      
      // Backend audio devices  
      audioInputs: this.backendAudioDevices.input,
      audioOutputs: this.backendAudioDevices.output,
      
      // Selection status
      hasCamera: this.webDevices.cameras.length > 0,
      hasMicrophone: this.webDevices.microphones.length > 0,
      hasAudioInput: this.backendAudioDevices.input.length > 0,
      hasAudioOutput: this.backendAudioDevices.output.length > 0
    }
  }

  /**
   * Get recommended device configurations
   * @returns {Object} Recommended device setup
   */
  getRecommendedDevices() {
    const devices = this.getAllDevices()
    
    // Smart device selection based on available devices
    const recommendations = {
      // For cameras, prefer external cameras over built-in
      audienceCamera: this._selectBestCamera(devices.cameras, 'external'),
      artistCamera: this._selectBestCamera(devices.cameras, 'any'),
      
      // For microphones, prefer USB/external over built-in
      artistMicrophone: this._selectBestMicrophone(devices.microphones),
      
      // For audio, prefer dedicated audio interfaces
      audioInput: this._selectBestAudioInput(devices.audioInputs),
      audioOutput: this._selectBestAudioOutput(devices.audioOutputs)
    }
    
    return recommendations
  }

  /**
   * Select devices and validate configuration
   * @param {Object} deviceSelection - Selected device IDs
   * @returns {Promise<boolean>} Configuration success
   */
  async selectDevices(deviceSelection) {
    console.log('🎯 Selecting devices:', deviceSelection)
    
    try {
      // Validate device selection
      const validation = this._validateDeviceSelection(deviceSelection)
      if (!validation.valid) {
        throw new Error(`Invalid device selection: ${validation.errors.join(', ')}`)
      }
      
      // Update selected devices
      this.selectedDevices = {
        ...this.selectedDevices,
        ...deviceSelection
      }
      
      // Test device access
      const testResults = await this._testDeviceAccess()
      
      this._notifyListeners('selected', { 
        devices: this.selectedDevices,
        tests: testResults
      })
      
      return testResults.allPassed
    } catch (error) {
      console.error('❌ Device selection failed:', error)
      this._notifyListeners('error', { error: error.message })
      return false
    }
  }

  /**
   * Start audio processing with selected devices
   * @returns {Promise<boolean>} Start success
   */
  async startAudioProcessing() {
    if (!this.selectedDevices.audioInput || !this.selectedDevices.audioOutput) {
      throw new Error('Audio devices not selected')
    }
    
    console.log('🎧 Starting audio processing with:', {
      input: this.selectedDevices.audioInput,
      output: this.selectedDevices.audioOutput
    })
    
    try {
      // Map device selections to backend format
      const inputDevice = this._getBackendDeviceName(this.selectedDevices.audioInput, 'input')
      const outputDevice = this._getBackendDeviceName(this.selectedDevices.audioOutput, 'output')
      
      // Start backend audio processing
      const response = await backendAPI.startAudio(inputDevice, outputDevice)
      
      if (response?.success) {
        this._notifyListeners('audioStarted', { 
          input: inputDevice,
          output: outputDevice
        })
        return true
      } else {
        throw new Error('Backend audio start failed')
      }
    } catch (error) {
      console.error('❌ Audio processing start failed:', error)
      this._notifyListeners('error', { error: error.message })
      return false
    }
  }

  /**
   * Stop audio processing
   * @returns {Promise<boolean>} Stop success
   */
  async stopAudioProcessing() {
    try {
      const response = await backendAPI.stopAudio()
      
      if (response?.success) {
        this._notifyListeners('audioStopped')
        return true
      } else {
        throw new Error('Backend audio stop failed')
      }
    } catch (error) {
      console.error('❌ Audio processing stop failed:', error)
      return false
    }
  }

  /**
   * Get camera stream for preview
   * @param {string} cameraId - Camera device ID
   * @returns {Promise<MediaStream>} Camera stream
   */
  async getCameraStream(cameraId) {
    try {
      const constraints = {
        video: {
          deviceId: cameraId ? { exact: cameraId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 }
        }
      }
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      return stream
    } catch (error) {
      console.error('❌ Camera stream failed:', error)
      throw error
    }
  }

  /**
   * Subscribe to device manager events
   * @param {Function} callback - Event callback (event, data) => {}
   */
  onDeviceEvent(callback) {
    this.deviceListeners.push(callback)
  }

  /**
   * Unsubscribe from device manager events
   * @param {Function} callback - Event callback to remove
   */
  offDeviceEvent(callback) {
    const index = this.deviceListeners.indexOf(callback)
    if (index > -1) {
      this.deviceListeners.splice(index, 1)
    }
  }

  // Private methods

  /**
   * Request media permissions
   * @private
   */
  async _requestPermissions() {
    try {
      // Request camera and microphone permissions
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      })
      
      // Stop the stream immediately - we just needed permissions
      stream.getTracks().forEach(track => track.stop())
      
      console.log('✅ Media permissions granted')
    } catch (error) {
      console.warn('⚠️ Media permissions limited:', error.message)
      // Continue anyway - we'll get device info without labels
    }
  }

  /**
   * Refresh web API devices (cameras, microphones)
   * @private
   */
  async _refreshWebDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      
      this.webDevices.cameras = devices
        .filter(device => device.kind === 'videoinput')
        .map((device, index) => ({
          id: device.deviceId,
          name: device.label || `Camera ${index + 1}`,
          type: 'camera',
          kind: 'videoinput',
          groupId: device.groupId
        }))
      
      this.webDevices.microphones = devices
        .filter(device => device.kind === 'audioinput')
        .map((device, index) => ({
          id: device.deviceId,
          name: device.label || `Microphone ${index + 1}`,
          type: 'microphone',
          kind: 'audioinput',
          groupId: device.groupId
        }))
      
      this.webDevices.speakers = devices
        .filter(device => device.kind === 'audiooutput')
        .map((device, index) => ({
          id: device.deviceId,
          name: device.label || `Speaker ${index + 1}`,
          type: 'speaker',
          kind: 'audiooutput',
          groupId: device.groupId
        }))
      
      console.log('✅ Web devices refreshed:', {
        cameras: this.webDevices.cameras.length,
        microphones: this.webDevices.microphones.length,
        speakers: this.webDevices.speakers.length
      })
      
      return true
    } catch (error) {
      console.error('❌ Web device refresh failed:', error)
      return false
    }
  }

  /**
   * Refresh backend audio devices
   * @private
   */
  async _refreshBackendAudioDevices() {
    try {
      const devices = await backendAPI.getAvailableDevices()
      
      this.backendAudioDevices = {
        input: devices.input_devices || [],
        output: devices.output_devices || []
      }
      
      console.log('✅ Backend audio devices refreshed:', {
        inputs: this.backendAudioDevices.input.length,
        outputs: this.backendAudioDevices.output.length
      })
      
      return true
    } catch (error) {
      console.error('❌ Backend audio device refresh failed:', error)
      
      // Fallback to default devices
      this.backendAudioDevices = {
        input: [{ index: 0, name: 'Default Input', channels: 1 }],
        output: [{ index: 0, name: 'Default Output', channels: 2 }]
      }
      
      return false
    }
  }

  /**
   * Select best camera based on criteria
   * @private
   */
  _selectBestCamera(cameras, preference = 'any') {
    if (!cameras || cameras.length === 0) return null
    
    // Prefer external cameras (USB) over built-in
    if (preference === 'external') {
      const external = cameras.find(cam => 
        cam.name.toLowerCase().includes('usb') ||
        cam.name.toLowerCase().includes('external') ||
        cam.name.toLowerCase().includes('webcam')
      )
      if (external) return external.id
    }
    
    // Return first available camera
    return cameras[0]?.id || null
  }

  /**
   * Select best microphone
   * @private
   */
  _selectBestMicrophone(microphones) {
    if (!microphones || microphones.length === 0) return null
    
    // Prefer USB/external microphones
    const external = microphones.find(mic => 
      mic.name.toLowerCase().includes('usb') ||
      mic.name.toLowerCase().includes('external') ||
      mic.name.toLowerCase().includes('blue') ||
      mic.name.toLowerCase().includes('rode')
    )
    
    if (external) return external.id
    
    // Return first available microphone
    return microphones[0]?.id || null
  }

  /**
   * Select best audio input device
   * @private
   */
  _selectBestAudioInput(inputs) {
    if (!inputs || inputs.length === 0) return null
    
    // Prefer dedicated audio interfaces
    const audioInterface = inputs.find(input =>
      input.name.toLowerCase().includes('audio interface') ||
      input.name.toLowerCase().includes('focusrite') ||
      input.name.toLowerCase().includes('presonus') ||
      input.name.toLowerCase().includes('zoom')
    )
    
    if (audioInterface) return audioInterface.index
    
    // Return first available input
    return inputs[0]?.index || 0
  }

  /**
   * Select best audio output device
   * @private
   */
  _selectBestAudioOutput(outputs) {
    if (!outputs || outputs.length === 0) return null
    
    // Prefer studio monitors or audio interfaces
    const studioOutput = outputs.find(output =>
      output.name.toLowerCase().includes('studio') ||
      output.name.toLowerCase().includes('monitor') ||
      output.name.toLowerCase().includes('audio interface') ||
      output.name.toLowerCase().includes('focusrite') ||
      output.name.toLowerCase().includes('presonus')
    )
    
    if (studioOutput) return studioOutput.index
    
    // Return first available output
    return outputs[0]?.index || 0
  }

  /**
   * Validate device selection
   * @private
   */
  _validateDeviceSelection(selection) {
    const errors = []
    
    // Check required devices are selected
    if (!selection.audioInput && !selection.audioOutput) {
      errors.push('Audio input and output devices required')
    }
    
    // Validate device IDs exist
    if (selection.audienceCamera && !this.webDevices.cameras.find(c => c.id === selection.audienceCamera)) {
      errors.push('Invalid audience camera selection')
    }
    
    if (selection.artistCamera && !this.webDevices.cameras.find(c => c.id === selection.artistCamera)) {
      errors.push('Invalid artist camera selection')
    }
    
    if (selection.artistMicrophone && !this.webDevices.microphones.find(m => m.id === selection.artistMicrophone)) {
      errors.push('Invalid artist microphone selection')
    }
    
    if (selection.audioInput && !this.backendAudioDevices.input.find(i => i.index === selection.audioInput)) {
      errors.push('Invalid audio input selection')
    }
    
    if (selection.audioOutput && !this.backendAudioDevices.output.find(o => o.index === selection.audioOutput)) {
      errors.push('Invalid audio output selection')
    }
    
    return {
      valid: errors.length === 0,
      errors
    }
  }

  /**
   * Test device access
   * @private
   */
  async _testDeviceAccess() {
    const tests = {
      audienceCamera: false,
      artistCamera: false,
      artistMicrophone: false,
      allPassed: false
    }
    
    try {
      // Test camera access if selected
      if (this.selectedDevices.audienceCamera) {
        try {
          const stream = await this.getCameraStream(this.selectedDevices.audienceCamera)
          stream.getTracks().forEach(track => track.stop())
          tests.audienceCamera = true
        } catch (error) {
          console.warn('⚠️ Audience camera test failed:', error.message)
        }
      }
      
      // Test artist camera (if different from audience)
      if (this.selectedDevices.artistCamera && 
          this.selectedDevices.artistCamera !== this.selectedDevices.audienceCamera) {
        try {
          const stream = await this.getCameraStream(this.selectedDevices.artistCamera)
          stream.getTracks().forEach(track => track.stop())
          tests.artistCamera = true
        } catch (error) {
          console.warn('⚠️ Artist camera test failed:', error.message)
        }
      } else if (this.selectedDevices.artistCamera === this.selectedDevices.audienceCamera) {
        tests.artistCamera = tests.audienceCamera
      }
      
      // Test microphone access
      if (this.selectedDevices.artistMicrophone) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: { deviceId: { exact: this.selectedDevices.artistMicrophone } }
          })
          stream.getTracks().forEach(track => track.stop())
          tests.artistMicrophone = true
        } catch (error) {
          console.warn('⚠️ Microphone test failed:', error.message)
        }
      }
      
      // All tests pass if we have at least one working camera and audio devices are selected
      tests.allPassed = (tests.audienceCamera || tests.artistCamera) && 
                       (this.selectedDevices.audioInput !== null) &&
                       (this.selectedDevices.audioOutput !== null)
      
    } catch (error) {
      console.error('❌ Device testing failed:', error)
    }
    
    return tests
  }

  /**
   * Get backend device name from index
   * @private
   */
  _getBackendDeviceName(deviceIndex, type) {
    const devices = type === 'input' ? this.backendAudioDevices.input : this.backendAudioDevices.output
    const device = devices.find(d => d.index === deviceIndex)
    return device ? device.name : 'default'
  }

  /**
   * Notify device listeners
   * @private
   */
  _notifyListeners(event, data = {}) {
    this.deviceListeners.forEach(callback => {
      try {
        callback(event, data)
      } catch (error) {
        console.error('Error in device listener:', error)
      }
    })
  }
}

// Export singleton instance
const deviceManager = new DeviceManager()
export default deviceManager