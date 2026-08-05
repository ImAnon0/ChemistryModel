Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")

projectFolder = fileSystem.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = projectFolder

pythonPath = projectFolder & "\.venv\Scripts\pythonw.exe"
scriptPath = projectFolder & "\ChemistryModel.py"

shell.Run """" & pythonPath & """ """ & scriptPath & """", 0, False