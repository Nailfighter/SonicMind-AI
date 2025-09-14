import { useState, useEffect, useCallback } from 'react'
import socketService from '../services/SocketService'

/**
 * useSocket Hook - React hook for Socket.IO connection management
 * 
 * This hook provides a React-friendly interface to the SocketService,
 * managing connection state and providing methods for event handling.
 * 
 * @param {boolean} autoConnect - Whether to automatically connect on mount
 * @returns {Object} - Socket connection state and methods
 */
export const useSocket = (autoConnect = true) => {
  const [connectionState, setConnectionState] = useState('disconnected')
  const [connectionError, setConnectionError] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [reconnectAttempt, setReconnectAttempt] = useState(0)

  // Connection state handler
  const handleConnectionChange = useCallback((state, error = null, extra = null) => {
    console.log('🔄 Socket connection state changed:', state, error?.message, extra)
    
    setConnectionState(state)
    setConnectionError(error)
    
    switch (state) {
      case 'connected':
        setIsConnected(true)
        setIsConnecting(false)
        setReconnectAttempt(0)
        break
      case 'connecting':
        setIsConnected(false)
        setIsConnecting(true)
        break
      case 'reconnecting':
        setIsConnected(false)
        setIsConnecting(true)
        if (extra !== null) {
          setReconnectAttempt(extra)
        }
        break
      case 'disconnected':
        setIsConnected(false)
        setIsConnecting(false)
        setReconnectAttempt(0)
        break
      case 'error':
        setIsConnected(false)
        setIsConnecting(false)
        break
      default:
        break
    }
  }, [])

  // Connect to backend
  const connect = useCallback(async () => {
    console.log('🔌 useSocket: Initiating connection')
    try {
      const success = await socketService.connect()
      return success
    } catch (error) {
      console.error('🔌 useSocket: Connection failed', error)
      return false
    }
  }, [])

  // Disconnect from backend
  const disconnect = useCallback(() => {
    console.log('🔌 useSocket: Initiating disconnection')
    socketService.disconnect()
  }, [])

  // Emit event to backend
  const emit = useCallback(async (eventName, data = {}) => {
    try {
      const response = await socketService.emit(eventName, data)
      return response
    } catch (error) {
      console.error('📤 useSocket: Failed to emit event', eventName, error)
      throw error
    }
  }, [])

  // Subscribe to backend events
  const on = useCallback((eventName, callback) => {
    socketService.on(eventName, callback)
    
    // Return cleanup function
    return () => {
      socketService.off(eventName, callback)
    }
  }, [])

  // Unsubscribe from events
  const off = useCallback((eventName, callback = null) => {
    socketService.off(eventName, callback)
  }, [])

  // Get current status
  const getStatus = useCallback(() => {
    return socketService.getStatus()
  }, [])

  // Setup connection listener on mount
  useEffect(() => {
    console.log('🔌 useSocket: Setting up connection listener')
    
    // Subscribe to connection changes
    socketService.onConnectionChange(handleConnectionChange)
    
    // Auto-connect with delay if requested (staggered connections)
    if (autoConnect) {
      const delay = Math.random() * 3000 + 1000 // 1-4 second random delay
      console.log(`🔌 useSocket: Auto-connecting to backend in ${delay.toFixed(0)}ms`)
      const timeoutId = setTimeout(() => {
        connect().catch(err => {
          console.warn('Auto-connect failed:', err.message)
        })
      }, delay)
      
      // Cleanup timeout if component unmounts
      return () => {
        clearTimeout(timeoutId)
        console.log('🔌 useSocket: Cleaning up connection listener')
        socketService.offConnectionChange(handleConnectionChange)
      }
    }
    
    // Cleanup on unmount
    return () => {
      console.log('🔌 useSocket: Cleaning up connection listener')
      socketService.offConnectionChange(handleConnectionChange)
    }
  }, [autoConnect, connect, handleConnectionChange])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Don't disconnect on unmount - let the service manage connection lifecycle
      // socketService.cleanup() would disconnect all components using the hook
    }
  }, [])

  return {
    // Connection state
    isConnected,
    isConnecting,
    connectionState,
    connectionError,
    reconnectAttempt,
    
    // Connection methods
    connect,
    disconnect,
    
    // Event methods
    emit,
    on,
    off,
    
    // Utility methods
    getStatus
  }
}

/**
 * useSocketEvent Hook - Subscribe to specific Socket.IO events
 * 
 * This hook handles subscribing to socket events with automatic cleanup.
 * 
 * @param {string} eventName - The event name to listen for
 * @param {Function} callback - The callback function
 * @param {Array} deps - Dependencies array for callback
 */
export const useSocketEvent = (eventName, callback, deps = []) => {
  const { on } = useSocket(false) // Don't auto-connect for event-only hook
  
  useEffect(() => {
    if (!eventName || !callback) return
    
    console.log('👂 useSocketEvent: Subscribing to', eventName)
    const cleanup = on(eventName, callback)
    
    return cleanup
  }, [eventName, on, ...deps])
}

/**
 * useSocketEmit Hook - Emit events with loading state management
 * 
 * This hook provides a convenient way to emit events with loading state.
 * 
 * @returns {Object} - Emit function with loading state
 */
export const useSocketEmit = () => {
  const { emit } = useSocket(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  
  const emitWithLoading = useCallback(async (eventName, data = {}) => {
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await emit(eventName, data)
      setIsLoading(false)
      return response
    } catch (err) {
      setError(err)
      setIsLoading(false)
      throw err
    }
  }, [emit])
  
  return {
    emit: emitWithLoading,
    isLoading,
    error
  }
}

export default useSocket