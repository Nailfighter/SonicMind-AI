; Custom installer script for SonicMind-AI
; This script handles the Python backend integration

!macro preInit
  ; Create necessary directories
  CreateDirectory "$APPDATA\SonicMind-AI"
  CreateDirectory "$APPDATA\SonicMind-AI\models"
  CreateDirectory "$APPDATA\SonicMind-AI\logs"
!macroend

!macro customInstall
  ; Copy model files to user data directory
  CopyFiles "$INSTDIR\resources\fma_eq_model_npy.pth" "$APPDATA\SonicMind-AI\models\"
  
  ; Set up environment variables
  WriteRegStr HKCU "Environment" "SONICMIND_DATA_DIR" "$APPDATA\SonicMind-AI"
  WriteRegStr HKCU "Environment" "SONICMIND_MODEL_DIR" "$APPDATA\SonicMind-AI\models"
!macroend

!macro customUnInstall
  ; Clean up user data (optional - ask user first)
  MessageBox MB_YESNO "Do you want to remove all SonicMind-AI data and settings?" IDYES remove_data IDNO keep_data
  remove_data:
    RMDir /r "$APPDATA\SonicMind-AI"
  keep_data:
  
  ; Remove environment variables
  DeleteRegValue HKCU "Environment" "SONICMIND_DATA_DIR"
  DeleteRegValue HKCU "Environment" "SONICMIND_MODEL_DIR"
!macroend
