import socketService from './SocketService'

/**
 * BackendAPI - Socket.IO-based API service replacing IPC
 * 
 * This service provides a clean API interface for communicating with the
 * SonicMind-AI backend using Socket.IO instead of Electron IPC.
 * All methods return Promises and handle connection errors gracefully.
 */
class BackendAPI {
  constructor() {
    this.isConnected = false
    this.connectionListeners = []
    this.pendingRequests = new Map() // Track pending requests
    this.requestTimeout = 15000 // 15 second timeout for requests
    
    // Subscribe to connection state changes
    socketService.onConnectionChange(this._handleConnectionChange.bind(this))
  }

  // Connection state management
  _handleConnectionChange(state) {
    this.isConnected = state === 'connected'
    
    // Notify listeners of connection state changes
    this.connectionListeners.forEach(callback => {
      try {
        callback(this.isConnected, state)
      } catch (error) {
        console.error('Error in connection listener:', error)
      }
    })
  }

  /**
   * Subscribe to connection state changes
   * @param {Function} callback - Called with (isConnected, state)
   */
  onConnectionChange(callback) {
    this.connectionListeners.push(callback)
    // Immediately call with current state
    callback(this.isConnected, socketService.getStatus().connected ? 'connected' : 'disconnected')
  }

  /**
   * Remove connection state listener
   */
  offConnectionChange(callback) {
    const index = this.connectionListeners.indexOf(callback)
    if (index > -1) {
      this.connectionListeners.splice(index, 1)
    }
  }

  // System Information Methods
  
  /**
   * Get current system status from backend
   * @returns {Promise<Object>} System status object
   */
  async getSystemStatus() {
    return this._makeRequest('get_system_status', {}, 'Failed to get system status')
  }

  /**
   * Get available audio devices
   * @returns {Promise<Object>} Available devices object
   */
  async getAvailableDevices() {
    if (!this.isConnected) {
      throw new Error('Not connected to backend server')
    }
    
    try {
      return await socketService.emit('get_available_devices', {})
    } catch (error) {
      console.error('Failed to get available devices:', error)
      throw error
    }
  }

  // Audio System Control Methods

  /**
   * Start audio processing
   * @param {string} inputDevice - Input device name or 'default'
   * @param {string} outputDevice - Output device name or 'default' 
   * @returns {Promise<Object>} Response object with success status
   */
  async startAudio(inputDevice = 'default', outputDevice = 'default') {
    if (!this.isConnected) {
      throw new Error('Not connected to backend server')
    }

    try {
      return await socketService.emit('start_audio', {
        input_device: inputDevice,
        output_device: outputDevice
      })
    } catch (error) {
      console.error('Failed to start audio:', error)
      throw error
    }
  }

  /**
   * Stop audio processing
   * @returns {Promise<Object>} Response object with success status
   */
  async stopAudio() {
    if (!this.isConnected) {
      throw new Error('Not connected to backend server')
    }

    try {
      return await socketService.emit('stop_audio', {})
    } catch (error) {
      console.error('Failed to stop audio:', error)
      throw error
    }
  }

  // EQ Control Methods

  /**
   * Start automatic EQ processing
   * @returns {Promise<Object>} Response object with success status
   */
  async startAutoEQ() {
    if (!this.isConnected) {
      throw new Error('Not connected to backend server')
    }

    try {
      return await socketService.emit('start_auto_eq', {})
    } catch (error) {
      console.error('Failed to start auto EQ:', error)
      throw error
    }
  }

  /**
   * Stop automatic EQ processing
   * @returns {Promise<Object>} Response object with success status
   */
  async stopAutoEQ() {
    if (!this.isConnected) {
      throw new Error('Not connected to backend server')
    }

    try {
      return await socketService.emit('stop_auto_eq', {})
    } catch (error) {
      console.error('Failed to stop auto EQ:', error)
      throw error
    }
  }

  /**
   * Manually update EQ band
   * @param {number} bandIndex - EQ band index (0-4)
   * @param {string} parameter - Parameter to update ('gain_db', 'freq', 'q')
   * @param {number} value - New parameter value
   * @returns {Promise<Object>} Response object with success status
   */
  async updateEQBand(bandIndex, parameter, value) {
    if (!this.isConnected) {
      throw new Error('Not connected to backend server')
    }

    try {
      return await socketService.emit('manual_eq_update', {
        band_index: bandIndex,
        parameter: parameter,
        value: value
      })
    } catch (error) {
      console.error('Failed to update EQ band:', error)
      throw error
    }
  }

  /**
   * Reset all EQ bands to flat
   * @returns {Promise<Object>} Response object with success status
   */
  async resetEQ() {
    if (!this.isConnected) {
      throw new Error('Not connected to backend server')
    }

    try {
      return await socketService.emit('reset_eq', {})
    } catch (error) {
      console.error('Failed to reset EQ:', error)
      throw error
    }
  }

  // Detection System Methods

  /**
   * Start camera-based detection systems
   * @param {number} cameraIndex - Camera index (default: 0)
   * @returns {Promise<Object>} Response object with success status
   */
  async startDetection(cameraIndex = 0) {
    if (!this.isConnected) {
      throw new Error('Not connected to backend server')
    }

    try {
      return await socketService.emit('start_detection', {
        camera_index: cameraIndex
      })
    } catch (error) {
      console.error('Failed to start detection:', error)
      throw error
    }
  }

  /**
   * Stop detection systems
   * @returns {Promise<Object>} Response object with success status
   */
  async stopDetection() {
    if (!this.isConnected) {
      throw new Error('Not connected to backend server')
    }

    try {
      return await socketService.emit('stop_detection', {})
    } catch (error) {
      console.error('Failed to stop detection:', error)
      throw error
    }
  }

  // Event Subscription Methods

  /**
   * Subscribe to real-time backend events
   * @param {string} eventName - Event name to listen for
   * @param {Function} callback - Event handler callback
   * @returns {Function} Unsubscribe function
   */
  on(eventName, callback) {
    return socketService.on(eventName, callback)
  }

  /**
   * Unsubscribe from backend events
   * @param {string} eventName - Event name
   * @param {Function} callback - Event handler callback (optional)
   */
  off(eventName, callback = null) {
    socketService.off(eventName, callback)
  }

  // Legacy IPC Replacement Methods (for backward compatibility)

  /**
   * Process audio file (replaces IPC process-audio)
   * Note: This will need to be implemented when backend supports file processing
   * @param {string} audioData - Base64 encoded audio data
   * @param {string} filename - Original filename
   * @param {string} processType - Processing type
   * @returns {Promise<Object>} Processing result
   */
  async processAudio(audioData, filename, processType) {
    if (!this.isConnected) {
      throw new Error('Not connected to backend server')
    }

    // For now, return a placeholder response
    // This would need to be implemented in the backend as a Socket.IO event
    console.warn('processAudio: Backend file processing not yet implemented via Socket.IO')
    
    return {
      success: false,
      error: 'File processing via Socket.IO not yet implemented',
      message: 'Please use real-time audio processing instead'
    }
  }

  /**
   * Get backend data (replaces IPC get-backend-data)
   * Maps old route-based system to new Socket.IO events
   * @param {string} route - Data route ('time', 'random', 'system', 'weather')
   * @returns {Promise<Object>} Backend data
   */
  async getBackendData(route = 'system') {
    if (!this.isConnected) {
      throw new Error('Not connected to backend server')
    }

    switch (route) {
      case 'system':
        return await this.getSystemStatus()
        
      case 'time':
        return {
          timestamp: Date.now(),
          iso: new Date().toISOString(),
          message: 'Current time from Socket.IO API'
        }
        
      case 'random':
        return {
          random: Math.random(),
          timestamp: Date.now(),
          message: 'Random number from Socket.IO API'
        }
        
      case 'weather':
        return {
          weather: 'sunny',
          temperature: '72°F',
          message: 'Mock weather data from Socket.IO API'
        }
        
      default:
        throw new Error(`Unknown route: ${route}`)
    }
  }

  // Convenience methods that mirror the old preload API

  /**
   * Get time data (backward compatibility)
   * @returns {Promise<Object>} Time data
   */
  async getTimeData() {
    return await this.getBackendData('time')
  }

  /**
   * Get random number (backward compatibility)
   * @returns {Promise<Object>} Random data
   */
  async getRandomNumber() {
    return await this.getBackendData('random')
  }

  /**
   * Get system info (backward compatibility)
   * @returns {Promise<Object>} System info
   */
  async getSystemInfo() {
    return await this.getBackendData('system')
  }

  /**
   * Get weather data (backward compatibility)
   * @returns {Promise<Object>} Weather data
   */
  async getWeatherData() {
    return await this.getBackendData('weather')
  }

  // Connection utilities

  /**
   * Check if backend is connected
   * @returns {boolean} Connection status
   */
  isBackendConnected() {
    return this.isConnected
  }

  /**
   * Force connection to backend
   * @returns {Promise<boolean>} Connection success
   */
  async connect() {
    return await socketService.connect()
  }

  /**
   * Disconnect from backend
   */
  disconnect() {
    socketService.disconnect()
  }

  /**
   * Get connection status details
   * @returns {Object} Status details
   */
  getConnectionStatus() {
    return socketService.getStatus()
  }
  
  // Private bulletproof request method
  
  /**
   * Make a bulletproof request to the backend with timeout and retry
   * @private
   * @param {string} eventName - Socket.IO event name
   * @param {Object} data - Request data
   * @param {string} errorMessage - Error message prefix
   * @param {number} maxRetries - Maximum number of retries
   * @returns {Promise} Request result
   */
  async _makeRequest(eventName, data = {}, errorMessage = 'Request failed', maxRetries = 2) {
    if (!this.isConnected) {
      throw new Error('Not connected to backend server. Please check connection.')
    }
    
    const requestId = `${eventName}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    
    // Track pending request
    const requestPromise = this._executeRequestWithTimeout(eventName, data, requestId)
    this.pendingRequests.set(requestId, requestPromise)
    
    try {
      const result = await requestPromise
      return result
    } catch (error) {
      console.error(`${errorMessage}:`, error)
      
      // Retry logic for certain types of errors
      if (maxRetries > 0 && this._shouldRetryError(error)) {
        console.log(`🔄 Retrying ${eventName} (${maxRetries} attempts left)`)
        await this._delay(1000 * (3 - maxRetries)) // Progressive delay
        return this._makeRequest(eventName, data, errorMessage, maxRetries - 1)
      }
      
      throw new Error(`${errorMessage}: ${error.message}`)
    } finally {
      this.pendingRequests.delete(requestId)
    }
  }
  
  /**
   * Execute request with timeout
   * @private
   */
  async _executeRequestWithTimeout(eventName, data, requestId) {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error(`Request timeout after ${this.requestTimeout}ms`))
      }, this.requestTimeout)
      
      socketService.emit(eventName, data)
        .then(response => {
          clearTimeout(timeoutId)
          resolve(response)
        })
        .catch(error => {
          clearTimeout(timeoutId)
          reject(error)
        })
    })
  }
  
  /**
   * Check if error should trigger a retry
   * @private
   */
  _shouldRetryError(error) {
    const retryableErrors = [
      'timeout',
      'network',
      'connection',
      'disconnect'
    ]
    
    return retryableErrors.some(keyword => 
      error.message.toLowerCase().includes(keyword)
    )
  }
  
  /**
   * Delay utility
   * @private
   */
  _delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
  }
}

// Export singleton instance
const backendAPI = new BackendAPI()
export default backendAPI