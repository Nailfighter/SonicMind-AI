import React, { useState } from 'react'
import './assets/main.css'
import ProfileSelector from './components/ProfileSelector.jsx'
import ProfilePage from './components/ProfilePage.jsx'
import UploadVideoPage from './components/UploadVideoPage.jsx'
import UploadSettingsPage from './components/UploadSettingsPage.jsx'
import DeviceSettingsPage from './components/DeviceSettingsPage.jsx'
import LoadingPage from './components/LoadingPage.jsx'
import LiveSettingPage from './components/LiveSettingPage.jsx'


const App = () => {
  const [selectedProfile, setSelectedProfile] = useState(null)
  const [currentPage, setCurrentPage] = useState('profiles') // 'profiles', 'profile', 'upload', 'upload-settings', 'devices', 'loading', or 'live'
  const [selectedMode, setSelectedMode] = useState(null)
  const [deviceSettings, setDeviceSettings] = useState(null)
  const [uploadData, setUploadData] = useState(null)
  
  // Sample profiles for demonstration
  const profiles = [
    { id: 1, name: 'Alex' },
    { id: 2, name: 'Jordan' }
  ]

  const handleProfileSelect = (profile) => {
    setSelectedProfile(profile)
    setCurrentPage('profile')
    console.log('Selected profile:', profile)
  }

  const handleCreateNew = () => {
    console.log('Create new profile')
    // Add logic to create new profile
  }

  const handleBackToProfiles = () => {
    console.log('Back to profiles clicked')
    setCurrentPage('profiles')
    setSelectedProfile(null)
    setSelectedMode(null)
    setDeviceSettings(null)
  }

  const handleModeSelect = (mode) => {
    setSelectedMode(mode)
    if (mode === 'upload') {
      setCurrentPage('upload')
    } else if (mode === 'live') {
      setCurrentPage('devices')
    }
  }

  const handleDeviceSettingsComplete = (settings) => {
    setDeviceSettings(settings)
    console.log('Device settings saved:', settings)
    setCurrentPage('loading')
  }

  const handleLoadingComplete = () => {
    setCurrentPage('live')
  }

  const handleBackToProfile = () => {
    console.log('Back to profile clicked - current page:', currentPage)
    setCurrentPage('profile')
    setSelectedMode(null)
    console.log('After setting - current page should be: profile')
  }

  const handleBackToDevices = () => {
    setCurrentPage('devices')
  }

  const handleUploadComplete = (uploadData) => {
    setUploadData(uploadData)
    setCurrentPage('upload-settings')
  }

  const handleUploadSettingsComplete = () => {
    setCurrentPage('loading')
  }

  const handleBackToUpload = () => {
    setCurrentPage('upload')
    setUploadData(null)
  }

  console.log('Current page:', currentPage, 'Selected profile:', selectedProfile)

  if (currentPage === 'upload' && selectedProfile) {
    return (
      <UploadVideoPage 
        profile={selectedProfile} 
        onBack={handleBackToProfile}
        onContinue={handleUploadComplete}
      />
    )
  }

  if (currentPage === 'upload-settings' && selectedProfile && uploadData) {
    return (
      <UploadSettingsPage 
        profile={selectedProfile} 
        uploadData={uploadData}
        onBack={handleBackToUpload}
        onContinue={handleUploadSettingsComplete}
      />
    )
  }

  if (currentPage === 'devices' && selectedProfile) {
    return (
      <DeviceSettingsPage 
        profile={selectedProfile} 
        onBack={handleBackToProfile}
        onContinue={handleDeviceSettingsComplete}
      />
    )
  }

  if (currentPage === 'loading' && selectedProfile) {
    return (
      <LoadingPage 
        onComplete={handleLoadingComplete}
      />
    )
  }

  if (currentPage === 'live' && selectedProfile) {
    return (
      <LiveSettingPage 
        profile={selectedProfile} 
        deviceSettings={deviceSettings}
        onBack={handleBackToDevices}
      />
    )
  }

  if (currentPage === 'profile' && selectedProfile) {
    return (
      <ProfilePage 
        profile={selectedProfile} 
        onBack={handleBackToProfiles}
        onModeSelect={handleModeSelect}
      />
    )
  }

  return (
    <div className="w-screen h-screen bg-gradient-to-br from-gray-50 to-gray-100 relative overflow-hidden">
      {/* Grid overlay */}
      <div 
        className="absolute inset-0 opacity-15"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0,0,0,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,0,0,0.1) 1px, transparent 1px)
          `,
          backgroundSize: '20px 20px'
        }}
      />
      
      {/* Main content */}
      <div className="relative z-10 w-full h-full flex items-center justify-center p-8">
        <ProfileSelector
          profiles={profiles}
          onProfileSelect={handleProfileSelect}
          onCreateNew={handleCreateNew}
        />
      </div>
    </div>
  )
}

export default App
