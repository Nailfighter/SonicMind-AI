import React from 'react'
import { useSocket } from '../hooks/useSocket'

/**
 * ConnectionStatus Component - Visual indicator for Socket.IO connection state
 * 
 * This component provides real-time feedback about the backend connection status
 * with appropriate visual indicators and error messages.
 */
const ConnectionStatus = ({ 
  position = 'top-right',
  showLabel = true,
  showError = true,
  className = ''
}) => {
  const { 
    isConnected, 
    isConnecting, 
    connectionState, 
    connectionError,
    reconnectAttempt,
    connect 
  } = useSocket(false) // Don't auto-connect from this component

  // Position classes
  const positionClasses = {
    'top-left': 'top-4 left-4',
    'top-right': 'top-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'bottom-right': 'bottom-4 right-4'
  }

  // Connection status styles
  const getStatusStyle = () => {
    switch (connectionState) {
      case 'connected':
        return {
          indicator: 'bg-green-500 shadow-green-200',
          text: 'text-green-600',
          label: 'Connected',
          icon: '🟢'
        }
      case 'connecting':
      case 'reconnecting':
        return {
          indicator: 'bg-yellow-500 shadow-yellow-200 animate-pulse',
          text: 'text-yellow-600',
          label: reconnectAttempt > 0 ? `Reconnecting (${reconnectAttempt})` : 'Connecting',
          icon: '🟡'
        }
      case 'error':
        return {
          indicator: 'bg-red-500 shadow-red-200',
          text: 'text-red-600',
          label: 'Connection Error',
          icon: '🔴'
        }
      case 'disconnected':
      default:
        return {
          indicator: 'bg-gray-400 shadow-gray-200',
          text: 'text-gray-600',
          label: 'Disconnected',
          icon: '⚪'
        }
    }
  }

  const status = getStatusStyle()

  const handleRetryConnection = async () => {
    if (!isConnecting && !isConnected) {
      await connect()
    }
  }

  return (
    <div className={`fixed ${positionClasses[position]} z-50 ${className}`}>
      <div className="flex items-center gap-2 bg-white rounded-lg shadow-lg border border-gray-200 px-3 py-2">
        {/* Status indicator */}
        <div className="relative">
          <div 
            className={`w-3 h-3 rounded-full ${status.indicator} shadow-sm`}
          />
          {/* Pulse animation for connecting state */}
          {(isConnecting || connectionState === 'reconnecting') && (
            <div 
              className={`absolute inset-0 w-3 h-3 rounded-full ${status.indicator.split(' ')[0]} animate-ping opacity-75`}
            />
          )}
        </div>

        {/* Status label */}
        {showLabel && (
          <span className={`text-sm font-medium ${status.text}`}>
            {status.label}
          </span>
        )}

        {/* Retry button for error state */}
        {connectionState === 'error' && (
          <button
            onClick={handleRetryConnection}
            className="text-xs bg-red-100 hover:bg-red-200 text-red-700 px-2 py-1 rounded transition-colors"
            disabled={isConnecting}
          >
            Retry
          </button>
        )}
      </div>

      {/* Error message */}
      {showError && connectionError && (
        <div className="mt-2 bg-red-50 border border-red-200 rounded-md p-2 max-w-sm">
          <p className="text-xs text-red-800 font-medium">Connection Error:</p>
          <p className="text-xs text-red-600 mt-1">
            {connectionError.message || 'Unable to connect to backend server'}
          </p>
          <p className="text-xs text-red-500 mt-1">
            Make sure the backend server is running on http://localhost:8000
          </p>
        </div>
      )}
    </div>
  )
}

/**
 * ConnectionStatusBadge - Inline connection status badge
 * 
 * A compact version of the connection status for embedding in other components.
 */
export const ConnectionStatusBadge = ({ className = '' }) => {
  const { isConnected, isConnecting, connectionState } = useSocket(false)
  
  const getBadgeStyle = () => {
    if (isConnected) {
      return 'bg-green-100 text-green-800 border-green-200'
    } else if (isConnecting) {
      return 'bg-yellow-100 text-yellow-800 border-yellow-200'
    } else {
      return 'bg-red-100 text-red-800 border-red-200'
    }
  }

  const getStatusText = () => {
    switch (connectionState) {
      case 'connected': return '✓ Connected'
      case 'connecting': return '⏳ Connecting'
      case 'reconnecting': return '🔄 Reconnecting'
      case 'error': return '✗ Error'
      default: return '○ Disconnected'
    }
  }

  return (
    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${getBadgeStyle()} ${className}`}>
      {getStatusText()}
    </span>
  )
}

/**
 * ConnectionGuard - Wrapper component that shows connection status
 * 
 * This component wraps other components and shows a connection overlay when disconnected.
 */
export const ConnectionGuard = ({ 
  children, 
  showOverlay = true,
  overlayMessage = 'Connecting to SonicMind-AI backend...',
  className = ''
}) => {
  const { isConnected, isConnecting, connectionError, connect } = useSocket()

  if (!isConnected && showOverlay) {
    return (
      <div className={`relative ${className}`}>
        {/* Render children with overlay */}
        <div className="filter blur-sm pointer-events-none opacity-50">
          {children}
        </div>
        
        {/* Connection overlay */}
        <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
          <div className="text-center p-6 bg-white rounded-lg shadow-lg border border-gray-200 max-w-md">
            {isConnecting ? (
              <>
                <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-800 mb-2">
                  {overlayMessage}
                </h3>
                <p className="text-sm text-gray-600">
                  Please wait while we establish the connection...
                </p>
              </>
            ) : (
              <>
                <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-red-600 text-xl">⚠</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-800 mb-2">
                  Backend Connection Required
                </h3>
                <p className="text-sm text-gray-600 mb-4">
                  Unable to connect to SonicMind-AI backend.
                </p>
                {connectionError && (
                  <p className="text-xs text-red-600 mb-4">
                    {connectionError.message}
                  </p>
                )}
                <button
                  onClick={connect}
                  className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded transition-colors text-sm"
                >
                  Retry Connection
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    )
  }

  return <div className={className}>{children}</div>
}

export default ConnectionStatus