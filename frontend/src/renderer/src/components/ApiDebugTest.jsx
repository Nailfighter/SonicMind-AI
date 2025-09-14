import React, { useState, useEffect } from 'react'
import apiService from '../services/apiService'

const ApiDebugTest = () => {
  const [connectionStatus, setConnectionStatus] = useState('Checking...')
  const [backendData, setBackendData] = useState(null)
  const [eqBands, setEqBands] = useState([])
  const [logs, setLogs] = useState([])
  
  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prev => [...prev, { timestamp, message, type }])
    console.log(`[${timestamp}] ${message}`)
  }
  
  const testConnection = async () => {
    addLog('Testing backend connection...', 'info')
    
    try {
      // Test health check
      const health = await apiService.healthCheck()
      if (health.success) {
        addLog('✅ Health check successful', 'success')
        setConnectionStatus('Connected')
        setBackendData(health.data)
        
        // Test EQ bands
        const eqResult = await apiService.getEQBands()
        if (eqResult.success) {
          addLog(`✅ EQ bands loaded: ${eqResult.data.length} bands`, 'success')
          setEqBands(eqResult.data)
        } else {
          addLog(`❌ EQ bands failed: ${eqResult.error}`, 'error')
        }
      } else {
        addLog(`❌ Health check failed: ${health.error}`, 'error')
        setConnectionStatus('Disconnected')
      }
    } catch (error) {
      addLog(`❌ Connection error: ${error.message}`, 'error')
      setConnectionStatus('Error')
    }
  }
  
  const testEQUpdate = async () => {
    addLog('Testing EQ band update...', 'info')
    
    try {
      const result = await apiService.updateEQBand(0, 'gain_db', 5.0)
      if (result.success) {
        addLog('✅ EQ update successful', 'success')
        // Refresh EQ bands
        const eqResult = await apiService.getEQBands()
        if (eqResult.success) {
          setEqBands(eqResult.data)
        }
      } else {
        addLog(`❌ EQ update failed: ${result.error}`, 'error')
      }
    } catch (error) {
      addLog(`❌ EQ update error: ${error.message}`, 'error')
    }
  }
  
  const testDirectFetch = async () => {
    addLog('Testing direct fetch...', 'info')
    
    try {
      const response = await fetch('http://localhost:8001/api/health')
      if (response.ok) {
        const data = await response.json()
        addLog('✅ Direct fetch successful', 'success')
        addLog(`Response: ${JSON.stringify(data)}`, 'info')
      } else {
        addLog(`❌ Direct fetch failed: ${response.status}`, 'error')
      }
    } catch (error) {
      addLog(`❌ Direct fetch error: ${error.message}`, 'error')
    }
  }
  
  useEffect(() => {
    testConnection()
  }, [])
  
  return (
    <div className="w-full h-screen bg-gray-100 p-4 overflow-y-auto">
      <div className="max-w-4xl mx-auto space-y-4">
        <h1 className="text-2xl font-bold text-gray-800">API Debug Test</h1>
        
        {/* Connection Status */}
        <div className="bg-white rounded-lg p-4 shadow">
          <h2 className="text-lg font-semibold mb-2">Connection Status</h2>
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${
              connectionStatus === 'Connected' ? 'bg-green-500' :
              connectionStatus === 'Disconnected' ? 'bg-red-500' : 'bg-yellow-500'
            }`} />
            <span>{connectionStatus}</span>
          </div>
        </div>
        
        {/* Test Buttons */}
        <div className="bg-white rounded-lg p-4 shadow">
          <h2 className="text-lg font-semibold mb-2">Tests</h2>
          <div className="space-x-2">
            <button 
              onClick={testConnection}
              className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
            >
              Test Connection
            </button>
            <button 
              onClick={testEQUpdate}
              className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded"
            >
              Test EQ Update
            </button>
            <button 
              onClick={testDirectFetch}
              className="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded"
            >
              Test Direct Fetch
            </button>
          </div>
        </div>
        
        {/* Backend Data */}
        {backendData && (
          <div className="bg-white rounded-lg p-4 shadow">
            <h2 className="text-lg font-semibold mb-2">Backend Data</h2>
            <pre className="bg-gray-100 p-2 rounded text-sm overflow-x-auto">
              {JSON.stringify(backendData, null, 2)}
            </pre>
          </div>
        )}
        
        {/* EQ Bands */}
        {eqBands.length > 0 && (
          <div className="bg-white rounded-lg p-4 shadow">
            <h2 className="text-lg font-semibold mb-2">EQ Bands</h2>
            <div className="space-y-2">
              {eqBands.map((band, index) => (
                <div key={index} className="flex justify-between items-center bg-gray-50 p-2 rounded">
                  <span>Band {index}: {band.freq}Hz</span>
                  <span>Gain: {band.gain}dB</span>
                  <span>Q: {band.q}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Logs */}
        <div className="bg-white rounded-lg p-4 shadow">
          <h2 className="text-lg font-semibold mb-2">Logs</h2>
          <div className="bg-black text-green-400 p-2 rounded font-mono text-sm h-64 overflow-y-auto">
            {logs.map((log, index) => (
              <div key={index} className={`${
                log.type === 'error' ? 'text-red-400' :
                log.type === 'success' ? 'text-green-400' : 'text-white'
              }`}>
                [{log.timestamp}] {log.message}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ApiDebugTest