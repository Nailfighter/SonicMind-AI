import { io } from 'socket.io-client'

/**
 * SocketService - Centralized Socket.IO client management
 * 
 * This service provides a singleton pattern for managing Socket.IO connections
 * with the SonicMind-AI backend. It handles connection state, event subscriptions,
 * error handling, and automatic reconnection.
 */
class SocketService {
  constructor() {
    this.socket = null
    this.isConnected = false
    this.isConnecting = false
    this.connectionError = null
    this.eventListeners = new Map()
    this.connectionListeners = []
    
    // Connection management
    this.connectionAttempts = 0
    this.maxConcurrentConnections = 1  // Only allow one connection at a time
    this.activeConnections = 0
    this.connectionQueue = []
    this.lastConnectionAttempt = 0
    this.minConnectionInterval = 2000  // 2 seconds between attempts
    
    // Rate limiting for events
    this.eventQueue = []
    this.maxEventsPerSecond = 10
    this.eventRateLimit = 1000 / this.maxEventsPerSecond
    this.lastEventTime = 0
    
    // Backend server configuration
    this.serverUrl = 'http://localhost:8000'
    this.connectionOptions = {
      transports: ['websocket', 'polling'],
      timeout: 10000,  // Increased timeout
      forceNew: false,  // Reuse connections when possible
      reconnection: true,
      reconnectionDelay: 2000,  // Slower reconnection
      reconnectionDelayMax: 10000,  // Max 10 seconds
      maxReconnectionAttempts: 3,  // Fewer attempts
      autoConnect: false,
      // Additional stability options
      upgrade: true,
      rememberUpgrade: true,
      pingTimeout: 60000,
      pingInterval: 25000
    }
    
    // Cleanup timer
    this.cleanupTimer = null
    
    // Bind methods to prevent context issues
    this._processEventQueue = this._processEventQueue.bind(this)
    this._scheduleCleanup = this._scheduleCleanup.bind(this)
    
    // Start event queue processor
    this._startEventProcessor()
  }

  /**
   * Connect to the Socket.IO server with bulletproof connection management
   * @returns {Promise<boolean>} - True if connection successful
   */
  async connect() {
    // Prevent multiple concurrent connections
    if (this.activeConnections >= this.maxConcurrentConnections) {
      console.warn('🚫 Maximum concurrent connections reached, queuing request')
      return new Promise((resolve) => {
        this.connectionQueue.push(resolve)
      })
    }
    
    if (this.isConnected) {
      console.log('✅ Already connected to backend')
      return true
    }
    
    if (this.isConnecting) {
      console.log('⏳ Connection already in progress')
      return this._waitForConnection()
    }

    // Rate limiting: ensure minimum interval between connection attempts
    const now = Date.now()
    const timeSinceLastAttempt = now - this.lastConnectionAttempt
    if (timeSinceLastAttempt < this.minConnectionInterval) {
      const delay = this.minConnectionInterval - timeSinceLastAttempt
      console.log(`⏱️ Rate limiting: waiting ${delay}ms before connection attempt`)
      await this._delay(delay)
    }
    
    this.lastConnectionAttempt = Date.now()
    this.connectionAttempts++
    this.activeConnections++
    
    console.log(`🔌 Connection attempt ${this.connectionAttempts} to SonicMind-AI backend at ${this.serverUrl}`)
    this.isConnecting = true
    this.connectionError = null

    try {
      // Clean up any existing socket before creating new one
      if (this.socket) {
        console.log('🧹 Cleaning up existing socket connection')
        this.socket.removeAllListeners()
        this.socket.disconnect()
        this.socket = null
        await this._delay(500) // Brief delay for cleanup
      }
      
      // Create socket instance with exponential backoff timeout
      const backoffTimeout = Math.min(
        this.connectionOptions.timeout + (this.connectionAttempts * 2000),
        30000 // Max 30 seconds
      )
      
      const connectionOptionsWithBackoff = {
        ...this.connectionOptions,
        timeout: backoffTimeout
      }
      
      console.log(`🔧 Creating socket with ${backoffTimeout}ms timeout`)
      this.socket = io(this.serverUrl, connectionOptionsWithBackoff)

      // Set up connection event handlers
      this._setupConnectionHandlers()

      // Connect to server
      this.socket.connect()

      // Wait for connection or timeout
      const connected = await this._waitForConnection()
      
      if (connected) {
        console.log('✅ Successfully connected to SonicMind-AI backend')
        this.isConnected = true
        this.isConnecting = false
        this.connectionAttempts = 0 // Reset on success
        this._notifyConnectionListeners('connected')
        this._processConnectionQueue()
        return true
      } else {
        throw new Error('Connection timeout')
      }

    } catch (error) {
      console.error(`❌ Connection attempt ${this.connectionAttempts} failed:`, error.message)
      this.connectionError = error
      this.isConnecting = false
      this._notifyConnectionListeners('error', error)
      
      // Exponential backoff for reconnection
      if (this.connectionAttempts < this.connectionOptions.maxReconnectionAttempts) {
        const delay = Math.min(2000 * Math.pow(2, this.connectionAttempts - 1), 10000)
        console.log(`⏱️ Will retry connection in ${delay}ms (attempt ${this.connectionAttempts + 1})`)
        setTimeout(() => {
          if (!this.isConnected) {
            this.connect().catch(() => {}) // Silent retry
          }
        }, delay)
      }
      
      return false
    } finally {
      this.activeConnections = Math.max(0, this.activeConnections - 1)
    }
  }

  /**
   * Disconnect from the Socket.IO server
   */
  disconnect() {
    if (this.socket) {
      console.log('🔌 Disconnecting from SonicMind-AI backend')
      this.socket.disconnect()
      this.socket = null
    }
    
    this.isConnected = false
    this.isConnecting = false
    this.connectionError = null
    this._notifyConnectionListeners('disconnected')
  }

  /**
   * Emit an event to the server with rate limiting
   * @param {string} eventName - The event name
   * @param {*} data - The data to send
   * @returns {Promise} - Promise that resolves when event is sent
   */
  emit(eventName, data = {}) {
    if (!this.isConnected || !this.socket) {
      console.warn('⚠️ Cannot emit event - not connected to backend')
      return Promise.reject(new Error('Not connected to backend'))
    }

    // Add to event queue for rate limiting
    return new Promise((resolve, reject) => {
      this.eventQueue.push({
        eventName,
        data,
        resolve,
        reject,
        timestamp: Date.now()
      })
    })
  }

  /**
   * Subscribe to an event from the server
   * @param {string} eventName - The event name to listen for
   * @param {Function} callback - The callback function
   */
  on(eventName, callback) {
    if (!this.socket) {
      console.warn('⚠️ Cannot subscribe to event - no socket connection')
      return
    }

    console.log('👂 Subscribing to event:', eventName)
    
    // Store the listener for cleanup later
    if (!this.eventListeners.has(eventName)) {
      this.eventListeners.set(eventName, [])
    }
    this.eventListeners.get(eventName).push(callback)
    
    // Add listener to socket
    this.socket.on(eventName, callback)
  }

  /**
   * Unsubscribe from an event
   * @param {string} eventName - The event name
   * @param {Function} callback - The callback function (optional)
   */
  off(eventName, callback = null) {
    if (!this.socket) return

    if (callback) {
      // Remove specific listener
      this.socket.off(eventName, callback)
      
      // Remove from our tracking
      const listeners = this.eventListeners.get(eventName)
      if (listeners) {
        const index = listeners.indexOf(callback)
        if (index > -1) {
          listeners.splice(index, 1)
        }
      }
    } else {
      // Remove all listeners for this event
      this.socket.off(eventName)
      this.eventListeners.delete(eventName)
    }

    console.log('👋 Unsubscribed from event:', eventName)
  }

  /**
   * Subscribe to connection state changes
   * @param {Function} callback - Callback function (state, error?) => {}
   */
  onConnectionChange(callback) {
    this.connectionListeners.push(callback)
    
    // Immediately call with current state
    if (this.isConnected) {
      callback('connected')
    } else if (this.isConnecting) {
      callback('connecting')
    } else if (this.connectionError) {
      callback('error', this.connectionError)
    } else {
      callback('disconnected')
    }
  }

  /**
   * Remove connection state listener
   * @param {Function} callback - The callback function to remove
   */
  offConnectionChange(callback) {
    const index = this.connectionListeners.indexOf(callback)
    if (index > -1) {
      this.connectionListeners.splice(index, 1)
    }
  }

  /**
   * Get current connection status
   * @returns {Object} - Status object with connection info
   */
  getStatus() {
    return {
      connected: this.isConnected,
      connecting: this.isConnecting,
      error: this.connectionError,
      serverUrl: this.serverUrl
    }
  }

  /**
   * Clean up all listeners and disconnect
   */
  cleanup() {
    console.log('🧹 Cleaning up SocketService')
    
    // Clear cleanup timer
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer)
      this.cleanupTimer = null
    }
    
    // Clear event queue
    this.eventQueue = []
    
    // Clear connection queue
    this.connectionQueue = []
    
    // Remove all event listeners
    for (const [eventName, listeners] of this.eventListeners) {
      this.socket?.off(eventName)
    }
    this.eventListeners.clear()
    
    // Clear connection listeners
    this.connectionListeners = []
    
    // Reset connection state
    this.connectionAttempts = 0
    this.activeConnections = 0
    this.lastConnectionAttempt = 0
    this.lastEventTime = 0
    
    // Disconnect
    this.disconnect()
  }
  
  // Bulletproof utility methods
  
  /**
   * Delay utility for rate limiting
   * @private
   */
  _delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
  }
  
  /**
   * Process connection queue
   * @private
   */
  _processConnectionQueue() {
    while (this.connectionQueue.length > 0 && this.isConnected) {
      const resolve = this.connectionQueue.shift()
      resolve(true)
    }
  }
  
  /**
   * Start event queue processor
   * @private
   */
  _startEventProcessor() {
    setInterval(this._processEventQueue, 100) // Process every 100ms
  }
  
  /**
   * Process event queue with rate limiting
   * @private
   */
  _processEventQueue() {
    if (!this.isConnected || !this.socket || this.eventQueue.length === 0) {
      return
    }
    
    const now = Date.now()
    
    // Remove expired events (older than 30 seconds)
    this.eventQueue = this.eventQueue.filter(event => {
      if (now - event.timestamp > 30000) {
        event.reject(new Error('Event expired due to rate limiting'))
        return false
      }
      return true
    })
    
    // Rate limiting check
    if (now - this.lastEventTime < this.eventRateLimit) {
      return
    }
    
    // Process next event in queue
    const event = this.eventQueue.shift()
    if (event) {
      this.lastEventTime = now
      console.log('📤 Processing queued event:', event.eventName, event.data)
      
      try {
        this.socket.emit(event.eventName, event.data, (response) => {
          console.log('📥 Received response for', event.eventName, response)
          event.resolve(response)
        })
      } catch (error) {
        console.error('❌ Error emitting event:', error)
        event.reject(error)
      }
    }
  }
  
  /**
   * Schedule periodic cleanup
   * @private
   */
  _scheduleCleanup() {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer)
    }
    
    this.cleanupTimer = setInterval(() => {
      // Clean up old event listeners
      for (const [eventName, listeners] of this.eventListeners) {
        if (listeners.length === 0) {
          this.eventListeners.delete(eventName)
        }
      }
      
      // Log connection stats
      if (this.eventQueue.length > 50) {
        console.warn(`⚠️ Event queue getting large: ${this.eventQueue.length} events`)
      }
      
      if (this.connectionQueue.length > 10) {
        console.warn(`⚠️ Connection queue getting large: ${this.connectionQueue.length} requests`)
      }
      
    }, 60000) // Every minute
  }

  // Private methods

  /**
   * Set up connection event handlers
   * @private
   */
  _setupConnectionHandlers() {
    this.socket.on('connect', () => {
      console.log('🔗 Socket connected to backend')
      this.isConnected = true
      this.isConnecting = false
      this.connectionError = null
      this._scheduleCleanup() // Start cleanup scheduler
    })

    this.socket.on('disconnect', (reason) => {
      console.log('💔 Socket disconnected:', reason)
      this.isConnected = false
      this.isConnecting = false
      this._notifyConnectionListeners('disconnected')
    })

    this.socket.on('connect_error', (error) => {
      console.error('❌ Socket connection error:', error.message)
      this.connectionError = error
      this.isConnecting = false
      this._notifyConnectionListeners('error', error)
    })

    this.socket.on('reconnect', (attemptNumber) => {
      console.log('🔄 Socket reconnected after', attemptNumber, 'attempts')
      this.isConnected = true
      this.connectionError = null
      this._notifyConnectionListeners('connected')
    })

    this.socket.on('reconnect_attempt', (attemptNumber) => {
      console.log('🔄 Socket reconnection attempt', attemptNumber)
      this._notifyConnectionListeners('reconnecting', null, attemptNumber)
    })

    this.socket.on('reconnect_failed', () => {
      console.error('❌ Socket reconnection failed - giving up')
      this.isConnected = false
      this.connectionError = new Error('Reconnection failed')
      this._notifyConnectionListeners('error', this.connectionError)
    })
  }

  /**
   * Wait for connection to complete
   * @private
   * @returns {Promise<boolean>}
   */
  _waitForConnection() {
    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        resolve(false)
      }, this.connectionOptions.timeout)

      const onConnect = () => {
        clearTimeout(timeout)
        this.socket.off('connect', onConnect)
        this.socket.off('connect_error', onError)
        resolve(true)
      }

      const onError = () => {
        clearTimeout(timeout)
        this.socket.off('connect', onConnect)
        this.socket.off('connect_error', onError)
        resolve(false)
      }

      this.socket.once('connect', onConnect)
      this.socket.once('connect_error', onError)
    })
  }

  /**
   * Notify connection listeners of state changes
   * @private
   */
  _notifyConnectionListeners(state, error = null, extra = null) {
    this.connectionListeners.forEach(callback => {
      try {
        callback(state, error, extra)
      } catch (err) {
        console.error('Error in connection listener:', err)
      }
    })
  }
}

// Export singleton instance
const socketService = new SocketService()
export default socketService