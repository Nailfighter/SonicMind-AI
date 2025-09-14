import { useState, useEffect, useCallback } from 'react'
import deviceManager from '../services/DeviceManager'

/**
 * useDeviceManager Hook - React hook for device management integration
 * 
 * This hook provides a React-friendly interface to the DeviceManager service,
 * managing device discovery, selection, and audio processing state.
 * 
 * @param {boolean} autoInitialize - Whether to auto-initialize on mount
 * @returns {Object} - Device management state and methods
 */
export const useDeviceManager = (autoInitialize = true) => {
  const [devices, setDevices] = useState({
    cameras: [],
    microphones: [],
    speakers: [],
    audioInputs: [],
    audioOutputs: [],
    hasCamera: false,
    hasMicrophone: false,
    hasAudioInput: false,
    hasAudioOutput: false
  })
  
  const [selectedDevices, setSelectedDevices] = useState({
    audienceCamera: null,
    artistCamera: null,
    artistMicrophone: null,
    audioInput: null,
    audioOutput: null
  })
  
  const [deviceTests, setDeviceTests] = useState({
    audienceCamera: false,
    artistCamera: false,
    artistMicrophone: false,
    allPassed: false
  })
  
  const [status, setStatus] = useState({
    isInitialized: false,
    isLoading: false,
    isRefreshing: false,
    audioProcessing: false,
    error: null
  })

  // Handle device manager events
  const handleDeviceEvent = useCallback((event, data) => {
    console.log('🎛️ Device event:', event, data)
    
    switch (event) {
      case 'initialized':
        setDevices(data.devices)
        setStatus(prev => ({ ...prev, isInitialized: true, isLoading: false }))
        break
        
      case 'refreshed':
        setDevices(data.devices)
        setStatus(prev => ({ ...prev, isRefreshing: false }))
        break
        
      case 'selected':
        setSelectedDevices(data.devices)
        setDeviceTests(data.tests)
        break
        
      case 'audioStarted':
        setStatus(prev => ({ ...prev, audioProcessing: true, error: null }))
        break
        
      case 'audioStopped':
        setStatus(prev => ({ ...prev, audioProcessing: false }))
        break
        
      case 'error':
        setStatus(prev => ({ ...prev, error: data.error, isLoading: false, isRefreshing: false }))
        break
        
      default:
        break
    }
  }, [])

  // Initialize device manager on mount
  useEffect(() => {
    if (autoInitialize) {
      setStatus(prev => ({ ...prev, isLoading: true, error: null }))
      
      deviceManager.initialize().catch(err => {
        setStatus(prev => ({ ...prev, error: err.message, isLoading: false }))
      })
    }
    
    // Subscribe to device events
    deviceManager.onDeviceEvent(handleDeviceEvent)
    
    return () => {
      deviceManager.offDeviceEvent(handleDeviceEvent)
    }
  }, [autoInitialize, handleDeviceEvent])

  // Device management methods
  const refreshDevices = useCallback(async (force = false) => {
    setStatus(prev => ({ ...prev, isRefreshing: true, error: null }))
    
    try {
      const success = await deviceManager.refreshDevices(force)
      if (!success) {
        throw new Error('Device refresh failed')
      }
    } catch (error) {
      setStatus(prev => ({ ...prev, error: error.message, isRefreshing: false }))
    }
  }, [])

  const selectDevices = useCallback(async (deviceSelection) => {
    setStatus(prev => ({ ...prev, error: null }))
    
    try {
      const success = await deviceManager.selectDevices(deviceSelection)
      return success
    } catch (error) {
      setStatus(prev => ({ ...prev, error: error.message }))
      return false
    }
  }, [])

  const getRecommendedDevices = useCallback(() => {
    return deviceManager.getRecommendedDevices()
  }, [])

  const startAudioProcessing = useCallback(async () => {
    setStatus(prev => ({ ...prev, error: null }))
    
    try {
      const success = await deviceManager.startAudioProcessing()
      return success
    } catch (error) {
      setStatus(prev => ({ ...prev, error: error.message }))
      return false
    }
  }, [])

  const stopAudioProcessing = useCallback(async () => {
    try {
      const success = await deviceManager.stopAudioProcessing()
      return success
    } catch (error) {
      setStatus(prev => ({ ...prev, error: error.message }))
      return false
    }
  }, [])

  const getCameraStream = useCallback(async (cameraId) => {
    try {
      return await deviceManager.getCameraStream(cameraId)
    } catch (error) {
      setStatus(prev => ({ ...prev, error: error.message }))
      throw error
    }
  }, [])

  // Device validation helpers
  const validateSelection = useCallback((selection) => {
    const requiredFields = ['audioInput', 'audioOutput']
    const missing = requiredFields.filter(field => !selection[field])
    
    if (missing.length > 0) {
      return {
        valid: false,
        errors: [`Missing required devices: ${missing.join(', ')}`]
      }
    }
    
    return { valid: true, errors: [] }
  }, [])

  const getDeviceById = useCallback((deviceId, type) => {
    switch (type) {
      case 'camera':
        return devices.cameras.find(d => d.id === deviceId)
      case 'microphone':
        return devices.microphones.find(d => d.id === deviceId)
      case 'audioInput':
        return devices.audioInputs.find(d => d.index === deviceId)
      case 'audioOutput':
        return devices.audioOutputs.find(d => d.index === deviceId)
      default:
        return null
    }
  }, [devices])

  // Device availability checks
  const hasRequiredDevices = devices.hasAudioInput && devices.hasAudioOutput
  const hasOptionalDevices = devices.hasCamera || devices.hasMicrophone
  const allDevicesSelected = selectedDevices.audioInput && selectedDevices.audioOutput

  return {
    // Device lists
    devices,
    selectedDevices,
    deviceTests,
    
    // Status
    status,
    hasRequiredDevices,
    hasOptionalDevices,
    allDevicesSelected,
    
    // Methods
    refreshDevices,
    selectDevices,
    getRecommendedDevices,
    startAudioProcessing,
    stopAudioProcessing,
    getCameraStream,
    validateSelection,
    getDeviceById,
    
    // Direct access to device manager
    deviceManager
  }
}

/**
 * useDeviceSelection Hook - Simplified device selection management
 * 
 * This hook provides a simpler interface for managing device selection state.
 * 
 * @param {Object} initialSelection - Initial device selection
 * @returns {Object} - Device selection state and methods
 */
export const useDeviceSelection = (initialSelection = {}) => {
  const [selection, setSelection] = useState({
    audienceCamera: null,
    artistCamera: null,
    artistMicrophone: null,
    audioInput: null,
    audioOutput: null,
    ...initialSelection
  })
  
  const updateDevice = useCallback((deviceType, deviceId) => {
    setSelection(prev => ({
      ...prev,
      [deviceType]: deviceId
    }))
  }, [])
  
  const updateMultipleDevices = useCallback((updates) => {
    setSelection(prev => ({
      ...prev,
      ...updates
    }))
  }, [])
  
  const resetSelection = useCallback(() => {
    setSelection({
      audienceCamera: null,
      artistCamera: null,
      artistMicrophone: null,
      audioInput: null,
      audioOutput: null,
      ...initialSelection
    })
  }, [initialSelection])
  
  const isComplete = useCallback((requiredDevices = ['audioInput', 'audioOutput']) => {
    return requiredDevices.every(device => selection[device] !== null)
  }, [selection])
  
  return {
    selection,
    updateDevice,
    updateMultipleDevices,
    resetSelection,
    isComplete
  }
}

/**
 * useCameraPreview Hook - Camera preview management
 * 
 * This hook manages camera stream for preview purposes.
 * 
 * @param {string} cameraId - Camera device ID
 * @returns {Object} - Camera preview state and methods
 */
export const useCameraPreview = (cameraId = null) => {
  const [stream, setStream] = useState(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState(null)
  
  const startPreview = useCallback(async (newCameraId = cameraId) => {
    if (!newCameraId) {
      setError('No camera ID provided')
      return false
    }
    
    try {
      setError(null)
      setIsStreaming(true)
      
      // Stop existing stream
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
      
      // Get new stream
      const newStream = await deviceManager.getCameraStream(newCameraId)
      setStream(newStream)
      
      return true
    } catch (err) {
      setError(err.message)
      setIsStreaming(false)
      return false
    }
  }, [cameraId, stream])
  
  const stopPreview = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
    setIsStreaming(false)
    setError(null)
  }, [stream])
  
  // Auto-start preview when camera ID changes
  useEffect(() => {
    if (cameraId && !stream) {
      startPreview(cameraId)
    }
  }, [cameraId, stream, startPreview])
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [stream])
  
  return {
    stream,
    isStreaming,
    error,
    startPreview,
    stopPreview
  }
}

export default useDeviceManager