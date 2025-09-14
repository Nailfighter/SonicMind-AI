/**
 * SonicMind AI - Frontend API Service
 * Handles all communication with the backend REST API
 */

import API_CONFIG from '../config/apiConfig.js'

class APIService {
  constructor() {
    this.baseURL = API_CONFIG.BASE_URL
    this.connected = false
    this.lastHealthCheck = null
    this.requestTimeout = API_CONFIG.TIMEOUTS.REQUEST
  }

  /**
   * Perform a health check to test backend connectivity
   */
  async healthCheck() {
    try {
      const response = await fetch(`${this.baseURL}/api/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (response.ok) {
        const data = await response.json()
        this.connected = true
        this.lastHealthCheck = Date.now()
        return { success: true, data }
      } else {
        this.connected = false
        return { success: false, error: 'Backend not responding' }
      }
    } catch (error) {
      this.connected = false
      return { success: false, error: error.message }
    }
  }

  /**
   * Get current system status
   */
  async getStatus() {
    try {
      const response = await fetch(`${this.baseURL}/api/status`)
      if (response.ok) {
        return { success: true, data: await response.json() }
      }
      return { success: false, error: 'Failed to get status' }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Get current EQ bands
   */
  async getEQBands() {
    try {
      const response = await fetch(`${this.baseURL}/api/eq/bands`)
      if (response.ok) {
        const data = await response.json()
        return { success: true, data: data.bands }
      }
      return { success: false, error: 'Failed to get EQ bands' }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Update a specific EQ band
   * @param {number} bandIndex - Index of the band (0-4)
   * @param {string} parameter - Parameter to update ('gain_db', 'freq', 'q')
   * @param {number} value - New value
   */
  async updateEQBand(bandIndex, parameter = 'gain_db', value) {
    try {
      const response = await fetch(`${this.baseURL}/api/eq/bands/${bandIndex}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          parameter,
          value
        })
      })

      if (response.ok) {
        const data = await response.json()
        return { success: true, data }
      }
      return { success: false, error: 'Failed to update EQ band' }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Reset all EQ bands to flat
   */
  async resetEQ() {
    try {
      const response = await fetch(`${this.baseURL}/api/eq/reset`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      })

      if (response.ok) {
        const data = await response.json()
        return { success: true, data }
      }
      return { success: false, error: 'Failed to reset EQ' }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Get recent events (for polling)
   * @param {number} since - Timestamp to get events since
   */
  async getEvents(since = 0) {
    try {
      const response = await fetch(`${this.baseURL}/api/events?since=${since}`)
      if (response.ok) {
        return { success: true, data: await response.json() }
      }
      return { success: false, error: 'Failed to get events' }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Start audio processing
   * @param {number} inputDevice - Input device index
   * @param {number} outputDevice - Output device index
   */
  async startAudio(inputDevice = null, outputDevice = null) {
    try {
      const response = await fetch(`${this.baseURL}/api/audio/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input_device: inputDevice,
          output_device: outputDevice
        })
      })

      if (response.ok) {
        return { success: true, data: await response.json() }
      }
      return { success: false, error: 'Failed to start audio' }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Stop audio processing
   */
  async stopAudio() {
    try {
      const response = await fetch(`${this.baseURL}/api/audio/stop`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      })

      if (response.ok) {
        return { success: true, data: await response.json() }
      }
      return { success: false, error: 'Failed to stop audio' }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Start auto-EQ
   */
  async startAutoEQ() {
    try {
      const response = await fetch(`${this.baseURL}/api/auto-eq/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      })

      if (response.ok) {
        return { success: true, data: await response.json() }
      }
      return { success: false, error: 'Failed to start auto-EQ' }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Stop auto-EQ
   */
  async stopAutoEQ() {
    try {
      const response = await fetch(`${this.baseURL}/api/auto-eq/stop`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      })

      if (response.ok) {
        return { success: true, data: await response.json() }
      }
      return { success: false, error: 'Failed to stop auto-EQ' }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Get available audio devices
   */
  async getDevices() {
    try {
      const response = await fetch(`${this.baseURL}/api/devices`)
      if (response.ok) {
        return { success: true, data: await response.json() }
      }
      return { success: false, error: 'Failed to get devices' }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Set current instrument
   * @param {string} instrument - Instrument name
   */
  async setInstrument(instrument) {
    try {
      const response = await fetch(`${this.baseURL}/api/instrument/set`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ instrument })
      })

      if (response.ok) {
        return { success: true, data: await response.json() }
      }
      return { success: false, error: 'Failed to set instrument' }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  /**
   * Check if backend is connected
   */
  isConnected() {
    return this.connected
  }
}

// Create and export a singleton instance
const apiService = new APIService()
export default apiService