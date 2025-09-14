import React, { useState, useEffect } from 'react'
import { useSocket } from '../hooks/useSocket'
import socketService from '../services/SocketService'

/**
 * ConnectionHealthMonitor - System health monitoring component
 * 
 * This component monitors the health of the Socket.IO connection and provides
 * bulletproof diagnostics and recovery options. Prevents system freezes by
 * monitoring connection patterns and resource usage.
 */
const ConnectionHealthMonitor = ({ 
  position = 'bottom-left',
  showDetails = false,
  className = '' 
}) => {
  const { isConnected, isConnecting, connectionState, reconnectAttempt } = useSocket(false)
  
  const [healthStats, setHealthStats] = useState({
    connectionAttempts: 0,
    activeConnections: 0,
    eventQueueLength: 0,
    connectionQueueLength: 0,
    lastSuccessfulConnection: null,
    connectionErrors: 0,
    systemStable: true
  })
  
  const [showMonitor, setShowMonitor] = useState(false)

  // Monitor connection health
  useEffect(() => {
    const monitorInterval = setInterval(() => {
      // Get internal stats from socketService
      const stats = {
        connectionAttempts: socketService.connectionAttempts || 0,
        activeConnections: socketService.activeConnections || 0,
        eventQueueLength: socketService.eventQueue?.length || 0,
        connectionQueueLength: socketService.connectionQueue?.length || 0,
        lastSuccessfulConnection: isConnected ? Date.now() : healthStats.lastSuccessfulConnection,
        connectionErrors: socketService.connectionError ? healthStats.connectionErrors + 1 : healthStats.connectionErrors,
        systemStable: determineSystemStability()
      }
      
      setHealthStats(stats)
    }, 1000) // Update every second

    return () => clearInterval(monitorInterval)
  }, [isConnected, healthStats.lastSuccessfulConnection, healthStats.connectionErrors])

  const determineSystemStability = () => {
    const queueThreshold = 20
    const attemptThreshold = 10
    
    return (
      healthStats.eventQueueLength < queueThreshold &&
      healthStats.connectionQueueLength < queueThreshold &&
      healthStats.connectionAttempts < attemptThreshold
    )
  }

  const getHealthStatus = () => {
    if (!healthStats.systemStable) {
      return { status: 'critical', color: 'red', icon: '🚨' }
    } else if (isConnecting || reconnectAttempt > 0) {
      return { status: 'warning', color: 'yellow', icon: '⚠️' }
    } else if (isConnected) {
      return { status: 'healthy', color: 'green', icon: '✅' }
    } else {
      return { status: 'disconnected', color: 'gray', icon: '❌' }
    }
  }

  const handleEmergencyReset = () => {
    console.log('🚨 Emergency connection reset triggered')
    
    // Emergency cleanup
    socketService.cleanup()
    
    // Wait a bit then attempt reconnection
    setTimeout(() => {
      socketService.connect().catch(err => {
        console.error('Emergency reconnect failed:', err)
      })
    }, 2000)
    
    // Reset health stats
    setHealthStats({
      connectionAttempts: 0,
      activeConnections: 0,
      eventQueueLength: 0,
      connectionQueueLength: 0,
      lastSuccessfulConnection: null,
      connectionErrors: 0,
      systemStable: true
    })
  }

  const handlePreventiveCleanup = () => {
    console.log('🧹 Preventive cleanup triggered')
    
    // Clear queues without full disconnect
    if (socketService.eventQueue) {
      socketService.eventQueue = []
    }
    if (socketService.connectionQueue) {
      socketService.connectionQueue = []
    }
  }

  const health = getHealthStatus()
  
  // Position classes
  const positionClasses = {
    'top-left': 'top-4 left-4',
    'top-right': 'top-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'bottom-right': 'bottom-4 right-4'
  }

  if (!showDetails && !showMonitor) {
    // Minimal health indicator
    return (
      <div 
        className={`fixed ${positionClasses[position]} z-40 ${className}`}
        onClick={() => setShowMonitor(true)}
      >
        <div className={`w-3 h-3 rounded-full cursor-pointer transition-all duration-300 ${
          health.status === 'healthy' ? 'bg-green-500' :
          health.status === 'warning' ? 'bg-yellow-500 animate-pulse' :
          health.status === 'critical' ? 'bg-red-500 animate-bounce' :
          'bg-gray-400'
        }`} title="Click to show connection health monitor" />
      </div>
    )
  }

  return (
    <div className={`fixed ${positionClasses[position]} z-40 ${className}`}>
      <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 min-w-64">
        {/* Header */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="text-lg">{health.icon}</span>
            <span className="text-sm font-medium text-gray-700">Connection Health</span>
          </div>
          <button
            onClick={() => setShowMonitor(false)}
            className="text-gray-400 hover:text-gray-600 text-sm"
          >
            ✕
          </button>
        </div>

        {/* Status */}
        <div className={`text-xs px-2 py-1 rounded mb-2 ${
          health.status === 'healthy' ? 'bg-green-100 text-green-800' :
          health.status === 'warning' ? 'bg-yellow-100 text-yellow-800' :
          health.status === 'critical' ? 'bg-red-100 text-red-800' :
          'bg-gray-100 text-gray-800'
        }`}>
          Status: {health.status.toUpperCase()}
        </div>

        {/* Health Metrics */}
        <div className="space-y-1 text-xs text-gray-600 mb-3">
          <div className="flex justify-between">
            <span>Connection State:</span>
            <span className="font-mono">{connectionState}</span>
          </div>
          <div className="flex justify-between">
            <span>Connection Attempts:</span>
            <span className={`font-mono ${healthStats.connectionAttempts > 5 ? 'text-orange-600' : ''}`}>
              {healthStats.connectionAttempts}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Active Connections:</span>
            <span className="font-mono">{healthStats.activeConnections}</span>
          </div>
          <div className="flex justify-between">
            <span>Event Queue:</span>
            <span className={`font-mono ${healthStats.eventQueueLength > 10 ? 'text-orange-600' : ''}`}>
              {healthStats.eventQueueLength}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Connection Queue:</span>
            <span className={`font-mono ${healthStats.connectionQueueLength > 5 ? 'text-orange-600' : ''}`}>
              {healthStats.connectionQueueLength}
            </span>
          </div>
          {reconnectAttempt > 0 && (
            <div className="flex justify-between">
              <span>Reconnect Attempt:</span>
              <span className="font-mono text-yellow-600">{reconnectAttempt}</span>
            </div>
          )}
        </div>

        {/* System Stability Warning */}
        {!healthStats.systemStable && (
          <div className="bg-red-50 border border-red-200 rounded p-2 mb-3">
            <div className="text-xs text-red-800 font-medium">⚠️ System Unstable</div>
            <div className="text-xs text-red-600 mt-1">
              High queue lengths or connection attempts detected. System may be overloaded.
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-1">
          {!healthStats.systemStable && (
            <button
              onClick={handleEmergencyReset}
              className="bg-red-500 hover:bg-red-600 text-white px-2 py-1 rounded text-xs transition-colors"
              title="Emergency reset - full cleanup and reconnect"
            >
              🚨 Reset
            </button>
          )}
          
          {(healthStats.eventQueueLength > 5 || healthStats.connectionQueueLength > 2) && (
            <button
              onClick={handlePreventiveCleanup}
              className="bg-orange-500 hover:bg-orange-600 text-white px-2 py-1 rounded text-xs transition-colors"
              title="Clear queues to prevent overload"
            >
              🧹 Clear
            </button>
          )}
          
          <button
            onClick={() => socketService.connect()}
            disabled={isConnected || isConnecting}
            className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white px-2 py-1 rounded text-xs transition-colors"
            title="Reconnect to backend"
          >
            🔌 Connect
          </button>
        </div>

        {/* Last Update */}
        <div className="text-xs text-gray-400 mt-2 text-center">
          Updated: {new Date().toLocaleTimeString()}
        </div>
      </div>
    </div>
  )
}

export default ConnectionHealthMonitor