# ChatterBox Setup - آسان طریقہ 🚀

## مسئلہ:
PyTorch 2.1.0 version available نہیں ہے

## حل:
Main ne 3 automated scripts بنائی ہیں جو سب کچھ automatically install کر دیں گی.

---

## طریقہ 1: PowerShell Script (Recommended)

### Step 1: PowerShell میں Execution Policy Set کریں

PowerShell کو **Administrator** mode میں کھولیں اور یہ command چلائیں:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 2: Setup Script چلائیں

```powershell
cd "C:\Users\tech solutionor\Desktop\newquemanagementinnextjs\que-management\python-tts-service"

.\setup.ps1
```

یہ script automatically:
- ✅ Virtual environment بنائے گی
- ✅ تمام dependencies install کرے گی
- ✅ PyTorch (latest version) install کرے گی

### Step 3: Service شروع کریں

```powershell
.\start.ps1
```

---

## طریقہ 2: Batch File (Easier - Double Click)

1. File Explorer میں جائیں:
   ```
   C:\Users\tech solutionor\Desktop\newquemanagementinnextjs\que-management\python-tts-service
   ```

2. `setup.bat` پر **double-click** کریں

3. Installation مکمل ہونے کا انتظار کریں (5-10 منٹ)

4. پھر `start.ps1` چلا کر service شروع کریں

---

## طریقہ 3: Manual Commands (اگر scripts کام نہیں کر رہیں)

```powershell
# Correct directory میں جائیں
cd "C:\Users\tech solutionor\Desktop\newquemanagementinnextjs\que-management\python-tts-service"

# Virtual environment activate کریں (اگر پہلے سے بنائی ہے)
.\venv\Scripts\Activate.ps1

# Dependencies ایک ایک کر کے install کریں
pip install flask==3.0.0
pip install flask-cors==4.0.0
pip install huggingface-hub==0.19.4
pip install transformers==4.35.0
pip install numpy==1.24.3
pip install scipy==1.11.4
pip install soundfile==0.12.1
pip install python-dotenv==1.0.0

# PyTorch install کریں (latest version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Service شروع کریں
python app.py
```

---

## ✅ کامیابی کی علامت:

جب service شروع ہو جائے تو آپ کو یہ message نظر آئے گا:

```
 * Running on http://0.0.0.0:5001
Press CTRL+C to quit
```

---

## ⚠️ عام غلطیاں:

### غلطی 1: Wrong Directory
```powershell
ERROR: Could not open requirements file
```
**حل:** Correct directory میں جائیں:
```powershell
cd "C:\Users\tech solutionor\Desktop\newquemanagementinnextjs\que-management\python-tts-service"
```

### غلطی 2: Virtual Environment Active نہیں ہے
```powershell
# پہلے activate کریں
.\venv\Scripts\Activate.ps1
```

### غلطی 3: Execution Policy Error
```powershell
# PowerShell Administrator mode میں:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## اگلا قدم:

Service start ہونے کے بعد:

1. Backend start کریں (دوسرے terminal میں):
   ```powershell
   cd backend
   npm start
   ```

2. Frontend start کریں (تیسرے terminal میں):
   ```powershell
   npm run dev
   ```

3. Browser میں کھولیں:
   ```
   http://localhost:3000/admin/configuration
   ```

---

**سوال؟** Documentation دیکھیں: `CHATTERBOX_INTEGRATION_GUIDE.md`
