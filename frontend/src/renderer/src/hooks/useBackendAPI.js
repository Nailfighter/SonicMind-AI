import { useState, useEffect, useCallback } from 'react'
import backendAPI from '../services/BackendAPI'

/**
 * useBackendAPI Hook - React hook for Backend API integration
 * 
 * This hook provides a React-friendly interface to the BackendAPI service,
 * managing connection state and providing easy access to backend methods.
 * It replaces the old IPC-based window.api calls.
 * 
 * @param {boolean} autoConnect - Whether to auto-connect on mount (default: true)
 * @returns {Object} - Backend API state and methods
 */
export const useBackendAPI = (autoConnect = true) => {
  const [isConnected, setIsConnected] = useState(false)
  const [connectionState, setConnectionState] = useState('disconnected')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // Connection state handler
  const handleConnectionChange = useCallback((connected, state) => {
    setIsConnected(connected)
    setConnectionState(state)
  }, [])

  // Setup connection listener
  useEffect(() => {
    backendAPI.onConnectionChange(handleConnectionChange)
    
    // Auto-connect if requested
    if (autoConnect) {
      backendAPI.connect().catch(err => {
        console.warn('Auto-connect failed:', err.message)
      })
    }
    
    return () => {
      backendAPI.offConnectionChange(handleConnectionChange)
    }
  }, [autoConnect, handleConnectionChange])

  // Generic API call wrapper with loading state
  const apiCall = useCallback(async (apiMethod, ...args) => {
    setIsLoading(true)
    setError(null)
    
    try {
      const result = await apiMethod(...args)
      setIsLoading(false)
      return result
    } catch (err) {
      setError(err)
      setIsLoading(false)
      throw err
    }
  }, [])

  // System Information Methods
  const getSystemStatus = useCallback(() => 
    apiCall(backendAPI.getSystemStatus.bind(backendAPI)), [apiCall])
  
  const getAvailableDevices = useCallback(() => 
    apiCall(backendAPI.getAvailableDevices.bind(backendAPI)), [apiCall])

  // Audio System Control
  const startAudio = useCallback((inputDevice, outputDevice) => 
    apiCall(backendAPI.startAudio.bind(backendAPI), inputDevice, outputDevice), [apiCall])
  
  const stopAudio = useCallback(() => 
    apiCall(backendAPI.stopAudio.bind(backendAPI)), [apiCall])

  // EQ Control
  const startAutoEQ = useCallback(() => 
    apiCall(backendAPI.startAutoEQ.bind(backendAPI)), [apiCall])
  
  const stopAutoEQ = useCallback(() => 
    apiCall(backendAPI.stopAutoEQ.bind(backendAPI)), [apiCall])
  
  const updateEQBand = useCallback((bandIndex, parameter, value) => 
    apiCall(backendAPI.updateEQBand.bind(backendAPI), bandIndex, parameter, value), [apiCall])
  
  const resetEQ = useCallback(() => 
    apiCall(backendAPI.resetEQ.bind(backendAPI)), [apiCall])

  // Detection System Control
  const startDetection = useCallback((cameraIndex) => 
    apiCall(backendAPI.startDetection.bind(backendAPI), cameraIndex), [apiCall])
  
  const stopDetection = useCallback(() => 
    apiCall(backendAPI.stopDetection.bind(backendAPI)), [apiCall])

  // Legacy IPC Replacement Methods
  const getBackendData = useCallback((route) => 
    apiCall(backendAPI.getBackendData.bind(backendAPI), route), [apiCall])
  
  const getTimeData = useCallback(() => 
    apiCall(backendAPI.getTimeData.bind(backendAPI)), [apiCall])
  
  const getRandomNumber = useCallback(() => 
    apiCall(backendAPI.getRandomNumber.bind(backendAPI)), [apiCall])
  
  const getSystemInfo = useCallback(() => 
    apiCall(backendAPI.getSystemInfo.bind(backendAPI)), [apiCall])
  
  const getWeatherData = useCallback(() => 
    apiCall(backendAPI.getWeatherData.bind(backendAPI)), [apiCall])
  
  const processAudio = useCallback((audioData, filename, processType) => 
    apiCall(backendAPI.processAudio.bind(backendAPI), audioData, filename, processType), [apiCall])

  // Connection Control
  const connect = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      const success = await backendAPI.connect()
      setIsLoading(false)
      return success
    } catch (err) {
      setError(err)
      setIsLoading(false)
      return false
    }
  }, [])

  const disconnect = useCallback(() => {
    backendAPI.disconnect()
  }, [])

  // Event subscription
  const on = useCallback((eventName, callback) => {
    return backendAPI.on(eventName, callback)
  }, [])

  const off = useCallback((eventName, callback) => {
    backendAPI.off(eventName, callback)
  }, [])

  return {
    // Connection state
    isConnected,
    connectionState,
    isLoading,
    error,
    
    // System information
    getSystemStatus,
    getAvailableDevices,
    
    // Audio system control
    startAudio,
    stopAudio,
    
    // EQ control
    startAutoEQ,
    stopAutoEQ,
    updateEQBand,
    resetEQ,
    
    // Detection system
    startDetection,
    stopDetection,
    
    // Legacy IPC replacements
    getBackendData,
    getTimeData,
    getRandomNumber,
    getSystemInfo,
    getWeatherData,
    processAudio,
    
    // Connection control
    connect,
    disconnect,
    
    // Event handling
    on,
    off,
    
    // Direct API access
    api: backendAPI
  }
}

/**
 * useBackendEvent Hook - Subscribe to specific backend events
 * 
 * This hook handles subscribing to backend events with automatic cleanup.
 * 
 * @param {string} eventName - The event name to listen for
 * @param {Function} callback - The callback function
 * @param {Array} deps - Dependencies array for callback
 */
export const useBackendEvent = (eventName, callback, deps = []) => {
  const { on } = useBackendAPI(false) // Don't auto-connect for event-only hook
  
  useEffect(() => {
    if (!eventName || !callback) return
    
    console.log('👂 useBackendEvent: Subscribing to', eventName)
    const cleanup = on(eventName, callback)
    
    return cleanup
  }, [eventName, on, ...deps])
}

/**
 * useBackendState Hook - Track backend state with real-time updates
 * 
 * This hook subscribes to backend system status and provides real-time state.
 * 
 * @returns {Object} - Backend system state
 */
export const useBackendState = () => {
  const { isConnected, getSystemStatus } = useBackendAPI()
  const [systemState, setSystemState] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)

  // Subscribe to system status updates
  useBackendEvent('system_status', (data) => {
    setSystemState(data)
    setLastUpdate(Date.now())
  })

  // Subscribe to EQ updates
  useBackendEvent('eq_updated', (data) => {
    setSystemState(prev => prev ? {
      ...prev,
      eq_bands: data.bands
    } : null)
    setLastUpdate(Date.now())
  })

  // Subscribe to detection updates
  useBackendEvent('instrument_detected', (data) => {
    setSystemState(prev => prev ? {
      ...prev,
      current_instrument: data.instrument
    } : null)
    setLastUpdate(Date.now())
  })

  // Initial system status fetch when connected
  useEffect(() => {
    if (isConnected && !systemState) {
      getSystemStatus().catch(err => {
        console.warn('Failed to get initial system status:', err.message)
      })
    }
  }, [isConnected, systemState, getSystemStatus])

  return {
    systemState,
    lastUpdate,
    isConnected
  }
}

export default useBackendAPI