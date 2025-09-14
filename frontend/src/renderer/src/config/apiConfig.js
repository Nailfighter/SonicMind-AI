/**
 * API Configuration for SonicMind AI
 * Centralized configuration for API endpoints and settings
 */

// Default API configuration
const API_CONFIG = {
  // Backend server configuration
  BACKEND: {
    HOST: 'localhost',
    PORT: 8001, // Updated to match the backend server port
    PROTOCOL: 'http'
  },
  
  // Polling intervals (in milliseconds)
  POLLING: {
    EVENTS: 1000,           // Poll for events every 1 second
    CONNECTION_CHECK: 10000, // Check connection every 10 seconds
    STATUS_UPDATE: 2000     // Update system status every 2 seconds
  },
  
  // Timeout settings (in milliseconds)
  TIMEOUTS: {
    REQUEST: 5000,          // 5 second timeout for requests
    CONNECTION: 3000        // 3 second timeout for connection checks
  },
  
  // EQ Configuration
  EQ: {
    BAND_COUNT: 5,
    MIN_GAIN: -12,
    MAX_GAIN: 12,
    DEFAULT_BANDS: [
      { freq: 80, gain: 0, q: 1.0 },
      { freq: 300, gain: 0, q: 1.2 },
      { freq: 1000, gain: 0, q: 1.5 },
      { freq: 4000, gain: 0, q: 2.0 },
      { freq: 10000, gain: 0, q: 1.0 }
    ]
  }
}

// Build the base URL
API_CONFIG.BASE_URL = `${API_CONFIG.BACKEND.PROTOCOL}://${API_CONFIG.BACKEND.HOST}:${API_CONFIG.BACKEND.PORT}`

// Environment-specific overrides
if (process.env.NODE_ENV === 'development') {
  // Development-specific settings
  API_CONFIG.POLLING.EVENTS = 500 // Faster polling in dev
}

export default API_CONFIG