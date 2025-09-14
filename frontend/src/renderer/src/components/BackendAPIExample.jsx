import React, { useState, useEffect } from 'react'
import { useBackendAPI, useBackendState } from '../hooks/useBackendAPI'

/**
 * BackendAPI Migration Example Component
 * 
 * This component demonstrates how to migrate from IPC-based backend communication
 * to the new Socket.IO-based BackendAPI service.
 */
const BackendAPIExample = () => {
  const {
    isConnected,
    isLoading,
    error,
    getSystemInfo,
    getTimeData,
    startAudio,
    stopAudio
  } = useBackendAPI()
  
  const { systemState } = useBackendState()
  
  const [exampleData, setExampleData] = useState(null)

  // Example: Fetching data on component mount
  useEffect(() => {
    if (isConnected) {
      // Old way (IPC):
      // window.api.getSystemInfo().then(setExampleData)
      
      // New way (BackendAPI):
      getSystemInfo()
        .then(setExampleData)
        .catch(err => console.error('Failed to get system info:', err))
    }
  }, [isConnected, getSystemInfo])

  const handleGetTimeData = async () => {
    try {
      // Old way (IPC):
      // const data = await window.api.getTimeData()
      
      // New way (BackendAPI):
      const data = await getTimeData()
      
      console.log('Time data:', data)
      alert(`Current time: ${data.iso}`)
    } catch (err) {
      console.error('Failed to get time:', err)
      alert(`Error: ${err.message}`)
    }
  }

  const handleAudioToggle = async () => {
    try {
      if (systemState?.audio_running) {
        await stopAudio()
        console.log('Audio stopped')
      } else {
        await startAudio()
        console.log('Audio started')
      }
    } catch (err) {
      console.error('Audio control failed:', err)
      alert(`Audio control error: ${err.message}`)
    }
  }

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg border border-gray-200 max-w-2xl mx-auto">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">
        🔄 IPC to BackendAPI Migration Example
      </h2>
      
      {/* Connection Status */}
      <div className="mb-4 p-3 bg-gray-50 rounded-lg">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Backend Connection:</span>
          <span className={`px-2 py-1 rounded text-sm ${
            isConnected 
              ? 'bg-green-100 text-green-800' 
              : 'bg-red-100 text-red-800'
          }`}>
            {isConnected ? '✅ Connected' : '❌ Disconnected'}
          </span>
        </div>
        
        {isLoading && (
          <div className="mt-2 text-sm text-blue-600">⏳ Loading...</div>
        )}
        
        {error && (
          <div className="mt-2 text-sm text-red-600">❌ {error.message}</div>
        )}
      </div>

      {/* System State Display */}
      {systemState && (
        <div className="mb-4 p-3 bg-blue-50 rounded-lg">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Real-time System State</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>Audio: {systemState.audio_running ? '🟢 Running' : '🔴 Stopped'}</div>
            <div>Auto-EQ: {systemState.auto_eq_running ? '🟢 Active' : '🔴 Inactive'}</div>
            <div>Detection: {systemState.detection_running ? '🟢 Running' : '🔴 Stopped'}</div>
            <div>Instrument: {systemState.current_instrument || 'None'}</div>
          </div>
        </div>
      )}

      {/* Example Data */}
      {exampleData && (
        <div className="mb-4 p-3 bg-green-50 rounded-lg">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Example Data (loaded on mount)</h3>
          <pre className="text-xs bg-white p-2 rounded border overflow-x-auto">
            {JSON.stringify(exampleData, null, 2)}
          </pre>
        </div>
      )}

      {/* Interactive Examples */}
      <div className="space-y-3">
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-2">Interactive Examples</h3>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={handleGetTimeData}
              disabled={!isConnected || isLoading}
              className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white px-3 py-2 rounded text-sm transition-colors"
            >
              Get Time Data
            </button>
            
            <button
              onClick={handleAudioToggle}
              disabled={!isConnected || isLoading}
              className={`px-3 py-2 rounded text-sm transition-colors text-white ${
                systemState?.audio_running
                  ? 'bg-red-500 hover:bg-red-600 disabled:bg-gray-300'
                  : 'bg-green-500 hover:bg-green-600 disabled:bg-gray-300'
              }`}
            >
              {systemState?.audio_running ? 'Stop Audio' : 'Start Audio'}
            </button>
          </div>
        </div>

        {/* Migration Code Examples */}
        <div className="border-t pt-4">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Code Migration Examples</h3>
          
          <div className="space-y-3">
            <div className="bg-red-50 border border-red-200 rounded p-3">
              <div className="text-xs font-medium text-red-800 mb-1">❌ OLD (IPC):</div>
              <code className="text-xs text-red-700 block">
                {`// Old IPC approach
const data = await window.api.getSystemInfo()
const timeData = await window.api.getTimeData()`}
              </code>
            </div>
            
            <div className="bg-green-50 border border-green-200 rounded p-3">
              <div className="text-xs font-medium text-green-800 mb-1">✅ NEW (BackendAPI):</div>
              <code className="text-xs text-green-700 block">
                {`// New BackendAPI approach
import { useBackendAPI } from '../hooks/useBackendAPI'

const { getSystemInfo, getTimeData } = useBackendAPI()
const systemData = await getSystemInfo()
const timeData = await getTimeData()`}
              </code>
            </div>
          </div>
        </div>

        {/* Benefits */}
        <div className="border-t pt-4">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Benefits of BackendAPI</h3>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>✅ Real-time events and updates</li>
            <li>✅ Better error handling</li>
            <li>✅ Connection state management</li>
            <li>✅ No Electron process spawning overhead</li>
            <li>✅ Direct Socket.IO communication</li>
            <li>✅ TypeScript-ready interfaces</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default BackendAPIExample