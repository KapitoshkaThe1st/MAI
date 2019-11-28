using System;
using System.Windows.Forms;
using System.ComponentModel;
using System.Xml.Serialization;
using System.Security.Permissions;
using System.Runtime.InteropServices;

namespace CGLabPlatform
{
    [Flags]
    public enum KeyMod : byte {
        None    = 0x00,
        Alt     = 0x01,
        Shift   = 0x02,
        Control = 0x04,
        Windows = 0x10
    }

    [SecurityPermission(SecurityAction.LinkDemand, Flags = SecurityPermissionFlag.UnmanagedCode)]
    public class Hotkey : IMessageFilter
    {
        public static void Register(Control control, KeyMod mod, Keys key, HandledEventHandler handler)
        {
            Hotkey hk   = new Hotkey();
            hk.KeyCode  = key;
            hk.Alt      = mod.HasFlag(KeyMod.Alt);
            hk.Shift    = mod.HasFlag(KeyMod.Shift);
            hk.Control  = mod.HasFlag(KeyMod.Control);
            hk.Windows  = mod.HasFlag(KeyMod.Windows);
            hk.Pressed  += handler;
            if (!hk.Register(control))
                throw new Exception("Ошибочка вышла - не удалось зарегестрировать клавишу");
        }

        #region Interop

        [DllImport("user32.dll", SetLastError = true)]
        private static extern int RegisterHotKey(IntPtr hWnd, int id, uint fsModifiers, Keys vk);

        [DllImport("user32.dll", SetLastError = true)]
        private static extern int UnregisterHotKey(IntPtr hWnd, int id);

        [DllImport("user32.dll")]
        private static extern IntPtr GetForegroundWindow();

        [DllImport("user32.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr SendMessageA(IntPtr hWnd, UInt32 Msg, IntPtr wParam, IntPtr lParam);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern IntPtr SendMessageW(IntPtr hWnd, UInt32 Msg, IntPtr wParam, IntPtr lParam);

        private const uint WM_HOTKEY = 0x312;
        private const uint MOD_ALT     = 0x1;
        private const uint MOD_CONTROL = 0x2;
        private const uint MOD_SHIFT   = 0x4;
        private const uint MOD_WIN     = 0x8;
        private const uint ERROR_HOTKEY_ALREADY_REGISTERED = 1409;

        #endregion

        private static int currentID;
        private const int maximumID = 0xBFFF;

        private Keys keyCode;
        private bool shift;
        private bool control;
        private bool alt;
        private bool windows;

        [XmlIgnore]
        private int id;
        [XmlIgnore]
        private bool registered;
        [XmlIgnore]
        private Control windowControl;

        public event HandledEventHandler Pressed;

        public Hotkey() : this(Keys.None, false, false, false, false) {}

        public Hotkey(Keys keyCode, bool shift, bool control, bool alt, bool windows) 
        {
            KeyCode = keyCode;
            Shift = shift;
            Control = control;
            Alt = alt;
            Windows = windows;

            Application.AddMessageFilter(this);
        }

        ~Hotkey() 
        {
            if (Registered)
                Unregister();
        }

        public Hotkey Clone() 
        {
            return new Hotkey(keyCode, shift, control, alt, windows);
        }

        public bool GetCanRegister(Control windowControl) 
        {
            try {
                if (!Register(windowControl))
                    return false;

                Unregister();
                return true;
            } catch (Win32Exception) {
                return false;
            } catch (NotSupportedException) {
                return false;
            }
        }

        public bool Register(Control windowControl) 
        {
            if (registered)
                throw new NotSupportedException("Невозможно зарегистрировать - горячая клавина уже использованна");

            if (Empty)
                throw new NotSupportedException("Невозможно зарегистрировать - горячая клавиша не заданна");

            id = Hotkey.currentID;
            Hotkey.currentID = Hotkey.currentID + 1 % Hotkey.maximumID;

            uint modifiers = (Alt ? Hotkey.MOD_ALT : 0) | (Control ? Hotkey.MOD_CONTROL : 0) |
                             (Shift ? Hotkey.MOD_SHIFT : 0) | (Windows ? Hotkey.MOD_WIN : 0);

            if (Hotkey.RegisterHotKey(windowControl.Handle, id, modifiers, keyCode) == 0) {
                if (Marshal.GetLastWin32Error() == ERROR_HOTKEY_ALREADY_REGISTERED) {
                    return false;
                } else
                    throw new Win32Exception();
            }

            registered = true;
            this.windowControl = windowControl;
            return true;
        }

        public void Unregister() 
        {
            if (!registered)
                throw new NotSupportedException("Невозможно удалить не зарегистрированную горячую клавиша");

            if (!windowControl.IsDisposed) {
                if (Hotkey.UnregisterHotKey(windowControl.Handle, id) == 0)
                    throw new Win32Exception();
            }

            registered = false;
            windowControl = null;
        }

        private void Reregister() 
        {
            if (!registered)
                return;

            Control windowControl = this.windowControl;

            Unregister();
            Register(windowControl);
        }

        public bool PreFilterMessage(ref Message message)
        {
            if (message.Msg != Hotkey.WM_HOTKEY)
                return false;

            var drawdevice = (Control)(windowControl as GDIDevice)
                                   ?? (windowControl as OGLDevice);
            if (drawdevice == null || !drawdevice.Focused)
                return false;

            if (registered && (message.WParam.ToInt32() == id)) {
                if (windowControl != null && !windowControl.ContainsFocus) {
                    SendMessageA(GetForegroundWindow(), 0x0100, (IntPtr)KeyCode, (IntPtr)(0));                 //KEYDOWN
                    SendMessageA(GetForegroundWindow(), 0x0101, (IntPtr)KeyCode, (IntPtr)(0x40000000));        //KEYUP
                    return false;
                }

                return OnPressed();
            } else
                return false;
        }

        private bool OnPressed() 
        {
            HandledEventArgs handledEventArgs = new HandledEventArgs(false);
            if (Pressed != null)
                Pressed(this, handledEventArgs);

            return handledEventArgs.Handled;
        }

        public override string ToString()
        {
            if (Empty)
                return "(none)";

            string keyName = Enum.GetName(typeof(Keys), keyCode); ;
            switch (keyCode) {
                case Keys.D0:
                case Keys.D1:
                case Keys.D2:
                case Keys.D3:
                case Keys.D4:
                case Keys.D5:
                case Keys.D6:
                case Keys.D7:
                case Keys.D8:
                case Keys.D9: keyName = keyName.Substring(1);
                              break;
            }

            string modifiers = "";
            if (shift)
                modifiers += "Shift+";
            if (control)
                modifiers += "Control+";
            if (alt)
                modifiers += "Alt+";
            if (windows)
                modifiers += "Windows+";

            return modifiers + keyName;
        }

        public bool Empty {
            get { return keyCode == Keys.None; }
        }

        public bool Registered {
            get { return registered; }
        }

        public Keys KeyCode {
            get { return keyCode; }
            set {
                keyCode = value;
                Reregister();
            }
        }

        public bool Shift {
            get { return this.shift; }
            set {
                shift = value;
                Reregister();
            }
        }

        public bool Control {
            get { return control; }
            set {
                control = value;
                Reregister();
            }
        }

        public bool Alt {
            get { return alt; }
            set {
                alt = value;
                Reregister();
            }
        }

        public bool Windows {
            get { return windows; }
            set {
                windows = value;
                Reregister();
            }
        }
    }
}
