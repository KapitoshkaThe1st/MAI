using System;
using System.Drawing;
using System.Windows.Forms;

namespace CGLabPlatform
{
    public class MouseExEventArgs
    {
        /// <summary>
        /// Возвращает константу, определяющуюу, зажатую кнопку мыши, связанную с этим событием.
        /// </summary>
        public MouseButtons PressedButton { get; private set; }

        /// <summary>
        /// Возвращает расположение указателя мыши в момент зажатия кнопки мыши, определяемой свойством PressedButton.
        /// </summary>
        public Point PressedLocation { get; private set; }

        /// <summary>
        /// Возвращает расположение указателя мыши в момент создания события мыши.
        /// </summary>
        public Point Location { get; private set; }

        /// <summary>
        /// Возвращает смещение указателя мыши по оси X в момент создания события мыши от расположения в котором была зажата кнопка мыши.
        /// </summary>
        public int DistanceX { get { return Location.X - PressedLocation.X; } }

        /// <summary>
        /// Возвращает смещение указателя мыши по оси Y в момент создания события мыши от расположения в котором была зажата кнопка мыши.
        /// </summary>
        public int DistanceY { get { return Location.Y - PressedLocation.Y; } }

        /// <summary>
        /// Возвращает изменение положения указателя мыши по оси X в момент создания события мыши.
        /// </summary>
        public int MovDeltaX { get { return Location.X - PrevLocation.X; } }

        /// <summary>
        /// Возвращает изменение положения указателя мыши по оси Y в момент создания события мыши.
        /// </summary>
        public int MovDeltaY { get { return Location.Y - PrevLocation.Y; } }

        /// <summary>
        /// Возвращает расстояние между указателем мыши в момент создания события мыши и расположением в котором была зажата кнопка мыши.
        /// </summary>
        public double Distance { get { return Math.Sqrt(DistanceX * DistanceX + DistanceY * DistanceY); } }

        /// <summary>
        /// Возвращает расстояние на которое изменилось положение указателя мыши в момент создания события мыши
        /// </summary>
        public double MovDelta { get { return Math.Sqrt(MovDeltaX * MovDeltaX + MovDeltaY * MovDeltaY); } }

        public double RotAngle
        {
            get
            {
                return 180.0 / Math.PI * (Math.Atan2(X - PressedLocation.X, Y - PressedLocation.Y) -
                                      Math.Atan2(PrevLocation.X - PressedLocation.X, PrevLocation.Y - PressedLocation.Y));
            }
        }

        /// <summary>
        /// Возвращает координату X указателя мыши в момент создания события мыши.
        /// </summary>
        public int X { get { return Location.X; } }

        /// <summary>
        /// Возвращает координату Y указателя мыши в момент создания события мыши.
        /// </summary>
        public int Y { get { return Location.Y; } }

        public double GetRotAngle(int cx, int cy)
        {
            return 180.0 / Math.PI * (Math.Atan2(X - cx, Y - cy) -
                Math.Atan2(PrevLocation.X - cx, PrevLocation.Y - cy));
        }

        private Point PrevLocation;

        internal bool _IsPressed = false;

        internal MouseExEventArgs(MouseButtons btn)
        {
            PressedButton = btn;
        }

        internal void Init(MouseEventArgs args)
        {
            PressedLocation = PrevLocation = Location = args.Location;
            _IsPressed = true;
        }

        internal MouseExEventArgs Update(MouseEventArgs args)
        {
            PrevLocation = Location;
            Location = args.Location;
            return this;
        }
    }

    public interface IDeviceUpdateArgs { }

    [System.ComponentModel.DesignerCategory("")]
    public abstract class DrawDevice<A> : UserControl where A : IDeviceUpdateArgs
    {
        public virtual event EventHandler<A> DeviceUpdate;

        //public event EventHandler<Progress> Progress;

        /// <summary>
        /// Происходит при перемещении указателя мыши с зажатой левой кнопкой мыши по элементу управления.
        /// </summary>
        public event EventHandler<MouseExEventArgs> MouseMoveWithLeftBtnDown;

        /// <summary>
        /// Происходит при перемещении указателя мыши с зажатой средней кнопкой мыши по элементу управления.
        /// </summary>
        public event EventHandler<MouseExEventArgs> MouseMoveWithMiddleBtnDown;

        /// <summary>
        /// Происходит при перемещении указателя мыши с зажатой правой кнопкой мыши по элементу управления.
        /// </summary>
        public event EventHandler<MouseExEventArgs> MouseMoveWithRightBtnDown;

        /* мое */

        public event EventHandler<MouseExEventArgs> MouseLeftBtnDown;
        public event EventHandler<MouseExEventArgs> MouseRightBtnDown;

        public event EventHandler<MouseExEventArgs> MouseLeftBtnUp;
        public event EventHandler<MouseExEventArgs> MouseRightBtnUp;

        public event EventHandler<MouseExEventArgs> MouseMove;

        /* /мое */

        private MouseExEventArgs _argsLBM = new MouseExEventArgs(MouseButtons.Left);
        private MouseExEventArgs _argsMMB = new MouseExEventArgs(MouseButtons.Middle);
        private MouseExEventArgs _argsRMB = new MouseExEventArgs(MouseButtons.Right);
        private MouseExEventArgs _argsEMPT = new MouseExEventArgs(MouseButtons.None);

        protected override void OnMouseDown(MouseEventArgs e)
        {
            switch (e.Button)
            {
                case MouseButtons.Left: _argsLBM.Init(e); MouseLeftBtnDown(this, _argsLBM.Update(e)); break;
                case MouseButtons.Middle: _argsMMB.Init(e); break;
                case MouseButtons.Right: _argsRMB.Init(e); MouseRightBtnDown(this, _argsRMB.Update(e)); break;
            }
            base.OnMouseDown(e);
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            switch (e.Button)
            {
                case MouseButtons.Left: _argsLBM._IsPressed = false; MouseLeftBtnUp(this, _argsLBM.Update(e)); break;
                case MouseButtons.Middle: _argsMMB._IsPressed = false; break;
                case MouseButtons.Right: _argsRMB._IsPressed = false; MouseRightBtnUp(this, _argsRMB.Update(e)); break;
            }
            base.OnMouseUp(e);
        }

        protected override void OnMouseEnter(EventArgs e)
        {
            base.OnMouseEnter(e);
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            MouseMove(this, _argsEMPT.Update(e));
            if (_argsLBM._IsPressed && null != MouseMoveWithLeftBtnDown)
                MouseMoveWithLeftBtnDown(this, _argsLBM.Update(e));
            if (_argsMMB._IsPressed && null != MouseMoveWithMiddleBtnDown)
                MouseMoveWithMiddleBtnDown(this, _argsMMB.Update(e));
            if (_argsRMB._IsPressed && null != MouseMoveWithRightBtnDown)
                MouseMoveWithRightBtnDown(this, _argsRMB.Update(e));

            base.OnMouseMove(e);
        }
    }
}
