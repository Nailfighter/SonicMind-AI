import React, { useState, useEffect } from 'react'
import { useSocket, useSocketEvent, useSocketEmit } from '../hooks/useSocket'
import { useBackendAPI, useBackendState } from '../hooks/useBackendAPI'

/**
 * SocketTest Component - Test Socket.IO connection and events
 * 
 * This component provides a test interface for verifying Socket.IO functionality
 * with the SonicMind-AI backend. It's useful for development and debugging.
 */
const SocketTest = () => {
  const { 
    isConnected, 
    isConnecting, 
    connectionState, 
    connectionError,
    connect, 
    disconnect,
    emit,
    on,
    off,
    getStatus 
  } = useSocket()
  
  const { emit: emitWithLoading, isLoading, error } = useSocketEmit()
  
  // Backend API hook for new Socket.IO-based API
  const {
    isConnected: backendConnected,
    getSystemStatus: backendGetSystemStatus,
    getTimeData,
    getRandomNumber,
    startAudio: backendStartAudio,
    stopAudio: backendStopAudio,
    resetEQ: backendResetEQ,
    error: backendError
  } = useBackendAPI(false) // Don't auto-connect since we're managing it manually
  
  const { systemState } = useBackendState()
  
  const [systemStatus, setSystemStatus] = useState(null)
  const [availableDevices, setAvailableDevices] = useState(null)
  const [testResults, setTestResults] = useState([])
  const [eventLog, setEventLog] = useState([])  
  const [backendTestResults, setBackendTestResults] = useState([])

  // Log events as they come in
  const logEvent = (eventName, data) => {
    const timestamp = new Date().toLocaleTimeString()
    setEventLog(prev => [
      ...prev.slice(-9), // Keep only last 10 events
      { timestamp, eventName, data }
    ])
  }

  // Subscribe to system events
  useSocketEvent('system_status', (data) => {
    console.log('📊 System status received:', data)
    setSystemStatus(data)
    logEvent('system_status', data)
  })

  useSocketEvent('available_devices', (data) => {
    console.log('🎧 Available devices received:', data)
    setAvailableDevices(data)
    logEvent('available_devices', data)
  })

  useSocketEvent('audio_started', (data) => {
    logEvent('audio_started', data)
  })

  useSocketEvent('audio_stopped', (data) => {
    logEvent('audio_stopped', data)
  })

  useSocketEvent('eq_updated', (data) => {
    logEvent('eq_updated', data)
  })

  useSocketEvent('instrument_detected', (data) => {
    logEvent('instrument_detected', data)
  })

  useSocketEvent('room_analysis', (data) => {
    logEvent('room_analysis', data)
  })

  // Test functions
  const testSystemStatus = async () => {
    try {
      const result = await emitWithLoading('get_system_status', {})
      addTestResult('get_system_status', true, result)
    } catch (err) {
      addTestResult('get_system_status', false, err.message)
    }
  }

  const testGetDevices = async () => {
    try {
      const result = await emitWithLoading('get_available_devices', {})
      addTestResult('get_available_devices', true, result)
    } catch (err) {
      addTestResult('get_available_devices', false, err.message)
    }
  }

  const testStartAudio = async () => {
    try {
      const result = await emitWithLoading('start_audio', {
        input_device: 'default',
        output_device: 'default'
      })
      addTestResult('start_audio', true, result)
    } catch (err) {
      addTestResult('start_audio', false, err.message)
    }
  }

  const testStopAudio = async () => {
    try {
      const result = await emitWithLoading('stop_audio', {})
      addTestResult('stop_audio', true, result)
    } catch (err) {
      addTestResult('stop_audio', false, err.message)
    }
  }

  const testResetEQ = async () => {
    try {
      const result = await emitWithLoading('reset_eq', {})
      addTestResult('reset_eq', true, result)
    } catch (err) {
      addTestResult('reset_eq', false, err.message)
    }
  }

  const addTestResult = (test, success, data) => {
    const timestamp = new Date().toLocaleTimeString()
    setTestResults(prev => [
      ...prev.slice(-9), // Keep only last 10 results
      { timestamp, test, success, data }
    ])
  }
  
  const addBackendTestResult = (test, success, data) => {
    const timestamp = new Date().toLocaleTimeString()
    setBackendTestResults(prev => [
      ...prev.slice(-9), // Keep only last 10 results
      { timestamp, test, success, data }
    ])
  }
  
  // BackendAPI test functions
  const testBackendSystemStatus = async () => {
    try {
      const result = await backendGetSystemStatus()
      addBackendTestResult('getSystemStatus', true, result)
    } catch (err) {
      addBackendTestResult('getSystemStatus', false, err.message)
    }
  }
  
  const testBackendTimeData = async () => {
    try {
      const result = await getTimeData()
      addBackendTestResult('getTimeData', true, result)
    } catch (err) {
      addBackendTestResult('getTimeData', false, err.message)
    }
  }
  
  const testBackendRandomNumber = async () => {
    try {
      const result = await getRandomNumber()
      addBackendTestResult('getRandomNumber', true, result)
    } catch (err) {
      addBackendTestResult('getRandomNumber', false, err.message)
    }
  }
  
  const testBackendStartAudio = async () => {
    try {
      const result = await backendStartAudio('default', 'default')
      addBackendTestResult('startAudio', true, result)
    } catch (err) {
      addBackendTestResult('startAudio', false, err.message)
    }
  }
  
  const testBackendStopAudio = async () => {
    try {
      const result = await backendStopAudio()
      addBackendTestResult('stopAudio', true, result)
    } catch (err) {
      addBackendTestResult('stopAudio', false, err.message)
    }
  }
  
  const testBackendResetEQ = async () => {
    try {
      const result = await backendResetEQ()
      addBackendTestResult('resetEQ', true, result)
    } catch (err) {
      addBackendTestResult('resetEQ', false, err.message)
    }
  }

  const clearLogs = () => {
    setEventLog([])
    setTestResults([])
    setBackendTestResults([])
  }

  if (!isConnected && !isConnecting) {
    return (
      <div className="p-6 bg-white rounded-lg shadow-lg border border-gray-200 max-w-4xl mx-auto">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          🔌 Socket.IO Test Interface
        </h2>
        <div className="text-center py-8">
          <p className="text-gray-600 mb-4">Not connected to backend</p>
          <button
            onClick={connect}
            className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded transition-colors"
          >
            Connect to Backend
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg border border-gray-200 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-800">
          🔌 Socket.IO Test Interface
        </h2>
        <div className="flex items-center gap-2">
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
            isConnected 
              ? 'bg-green-100 text-green-800' 
              : isConnecting 
                ? 'bg-yellow-100 text-yellow-800'
                : 'bg-red-100 text-red-800'
          }`}>
            {connectionState}
          </span>
          <button
            onClick={isConnected ? disconnect : connect}
            className={`px-3 py-1 rounded text-sm transition-colors ${
              isConnected 
                ? 'bg-red-500 hover:bg-red-600 text-white'
                : 'bg-blue-500 hover:bg-blue-600 text-white'
            }`}
            disabled={isConnecting}
          >
            {isConnected ? 'Disconnect' : 'Connect'}
          </button>
        </div>
      </div>

      {/* Connection Status */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Connection Details</h3>
        <div className="text-sm text-gray-600 space-y-1">
          <div>Server: http://localhost:8000</div>
          <div>State: {connectionState}</div>
          {connectionError && (
            <div className="text-red-600">Error: {connectionError.message}</div>
          )}
        </div>
      </div>

      {/* Test Buttons */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Event Tests</h3>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={testSystemStatus}
            disabled={!isConnected || isLoading}
            className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white px-3 py-2 rounded text-sm transition-colors"
          >
            Get System Status
          </button>
          <button
            onClick={testGetDevices}
            disabled={!isConnected || isLoading}
            className="bg-green-500 hover:bg-green-600 disabled:bg-gray-300 text-white px-3 py-2 rounded text-sm transition-colors"
          >
            Get Devices
          </button>
          <button
            onClick={testStartAudio}
            disabled={!isConnected || isLoading}
            className="bg-purple-500 hover:bg-purple-600 disabled:bg-gray-300 text-white px-3 py-2 rounded text-sm transition-colors"
          >
            Start Audio
          </button>
          <button
            onClick={testStopAudio}
            disabled={!isConnected || isLoading}
            className="bg-orange-500 hover:bg-orange-600 disabled:bg-gray-300 text-white px-3 py-2 rounded text-sm transition-colors"
          >
            Stop Audio
          </button>
          <button
            onClick={testResetEQ}
            disabled={!isConnected || isLoading}
            className="bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-300 text-white px-3 py-2 rounded text-sm transition-colors"
          >
            Reset EQ
          </button>
          <button
            onClick={clearLogs}
            className="bg-gray-500 hover:bg-gray-600 text-white px-3 py-2 rounded text-sm transition-colors"
          >
            Clear Logs
          </button>
        </div>
        {isLoading && (
          <div className="mt-2 text-sm text-blue-600">⏳ Sending request...</div>
        )}
        {error && (
          <div className="mt-2 text-sm text-red-600">❌ Error: {error.message}</div>
        )}
      </div>

      {/* Current System Status */}
      {systemStatus && (
        <div className="mb-6 p-4 bg-blue-50 rounded-lg">
          <h3 className="text-sm font-medium text-gray-700 mb-2">System Status</h3>
          <div className="text-sm space-y-1">
            <div>Audio Running: {systemStatus.audio_running ? '✅' : '❌'}</div>
            <div>Detection Running: {systemStatus.detection_running ? '✅' : '❌'}</div>
            <div>Auto-EQ Running: {systemStatus.auto_eq_running ? '✅' : '❌'}</div>
            <div>Current Instrument: {systemStatus.current_instrument}</div>
            <div>Room Preset: {systemStatus.room_preset}</div>
          </div>
        </div>
      )}

      {/* Available Devices */}
      {availableDevices && (
        <div className="mb-6 p-4 bg-green-50 rounded-lg">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Available Devices</h3>
          <div className="text-xs space-y-1 max-h-32 overflow-y-auto">
            {availableDevices.input && availableDevices.input.length > 0 && (
              <div>
                <strong>Input:</strong> {availableDevices.input.map(d => d.name).join(', ')}
              </div>
            )}
            {availableDevices.output && availableDevices.output.length > 0 && (
              <div>
                <strong>Output:</strong> {availableDevices.output.map(d => d.name).join(', ')}
              </div>
            )}
          </div>
        </div>
      )}

      {/* BackendAPI Test Section */}
      <div className="mb-6 p-4 bg-purple-50 rounded-lg border-2 border-purple-200">
        <h3 className="text-lg font-medium text-purple-800 mb-3">🆕 BackendAPI Tests (IPC Replacement)</h3>
        <p className="text-sm text-purple-600 mb-4">
          These tests use the new Socket.IO-based BackendAPI service instead of IPC.
          This is the new recommended way to communicate with the backend.
        </p>
        
        {/* Backend API Status */}
        <div className="mb-4 p-3 bg-white rounded border">
          <div className="flex items-center justify-between mb-2">
            <span className="font-medium text-gray-700">BackendAPI Status:</span>
            <span className={`px-2 py-1 rounded text-sm ${
              backendConnected 
                ? 'bg-green-100 text-green-800' 
                : 'bg-red-100 text-red-800'
            }`}>
              {backendConnected ? '✓ Connected' : '✗ Disconnected'}
            </span>
          </div>
          {systemState && (
            <div className="text-sm text-gray-600">
              <div>Audio: {systemState.audio_running ? '🎧 Running' : '⏸️ Stopped'}</div>
              <div>Auto-EQ: {systemState.auto_eq_running ? '🤖 Active' : '⏸️ Inactive'}</div>
              <div>Detection: {systemState.detection_running ? '📷 Running' : '⏸️ Stopped'}</div>
            </div>
          )}
        </div>
        
        {/* Backend API Test Buttons */}
        <div className="flex flex-wrap gap-2 mb-4">
          <button
            onClick={testBackendSystemStatus}
            disabled={!backendConnected}
            className="bg-purple-500 hover:bg-purple-600 disabled:bg-gray-300 text-white px-3 py-2 rounded text-sm transition-colors"
          >
            System Status
          </button>
          <button
            onClick={testBackendTimeData}
            disabled={!backendConnected}
            className="bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-300 text-white px-3 py-2 rounded text-sm transition-colors"
          >
            Time Data
          </button>
          <button
            onClick={testBackendRandomNumber}
            disabled={!backendConnected}
            className="bg-pink-500 hover:bg-pink-600 disabled:bg-gray-300 text-white px-3 py-2 rounded text-sm transition-colors"
          >
            Random Number
          </button>
          <button
            onClick={testBackendStartAudio}
            disabled={!backendConnected}
            className="bg-green-500 hover:bg-green-600 disabled:bg-gray-300 text-white px-3 py-2 rounded text-sm transition-colors"
          >
            Start Audio
          </button>
          <button
            onClick={testBackendStopAudio}
            disabled={!backendConnected}
            className="bg-red-500 hover:bg-red-600 disabled:bg-gray-300 text-white px-3 py-2 rounded text-sm transition-colors"
          >
            Stop Audio
          </button>
          <button
            onClick={testBackendResetEQ}
            disabled={!backendConnected}
            className="bg-yellow-500 hover:bg-yellow-600 disabled:bg-gray-300 text-white px-3 py-2 rounded text-sm transition-colors"
          >
            Reset EQ
          </button>
        </div>
        
        {backendError && (
          <div className="text-sm text-red-600 mb-2">❌ BackendAPI Error: {backendError.message}</div>
        )}
        
        {/* Backend API Test Results */}
        <div className="bg-white rounded border p-3">
          <h4 className="text-sm font-medium text-gray-700 mb-2">BackendAPI Test Results</h4>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {backendTestResults.length === 0 ? (
              <div className="text-xs text-gray-500">No BackendAPI test results yet</div>
            ) : (
              backendTestResults.map((result, index) => (
                <div key={index} className="text-xs p-2 bg-purple-50 rounded border">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-purple-700">{result.test}</span>
                    <span className={`px-1 py-0.5 rounded text-xs ${
                      result.success 
                        ? 'bg-green-100 text-green-700' 
                        : 'bg-red-100 text-red-700'
                    }`}>
                      {result.success ? '✓' : '✗'}
                    </span>
                  </div>
                  <div className="text-gray-500 mt-1">{result.timestamp}</div>
                  {result.data && (
                    <div className="mt-1 p-1 bg-gray-100 rounded text-xs font-mono max-h-16 overflow-y-auto">
                      {typeof result.data === 'string' ? result.data : JSON.stringify(result.data, null, 2)}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Test Results */}
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Test Results</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {testResults.length === 0 ? (
              <div className="text-sm text-gray-500">No test results yet</div>
            ) : (
              testResults.map((result, index) => (
                <div key={index} className="text-xs p-2 bg-white rounded border">
                  <div className="flex items-center justify-between">
                    <span className="font-mono">{result.test}</span>
                    <span className={`px-2 py-1 rounded text-xs ${
                      result.success 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {result.success ? '✓' : '✗'}
                    </span>
                  </div>
                  <div className="text-gray-500 mt-1">{result.timestamp}</div>
                  {result.data && (
                    <div className="mt-1 p-1 bg-gray-100 rounded text-xs font-mono">
                      {typeof result.data === 'string' ? result.data : JSON.stringify(result.data, null, 2)}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Event Log */}
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Event Log</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {eventLog.length === 0 ? (
              <div className="text-sm text-gray-500">No events yet</div>
            ) : (
              eventLog.map((event, index) => (
                <div key={index} className="text-xs p-2 bg-white rounded border">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-blue-600">{event.eventName}</span>
                    <span className="text-gray-500">{event.timestamp}</span>
                  </div>
                  {event.data && (
                    <div className="mt-1 p-1 bg-gray-100 rounded text-xs font-mono">
                      {JSON.stringify(event.data, null, 2)}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default SocketTest