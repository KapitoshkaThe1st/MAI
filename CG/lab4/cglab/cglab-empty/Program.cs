using System;
using System.Linq;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;
using System.ComponentModel;
using System.Collections.Generic;
using CGLabPlatform;

public abstract class CGLabEmpty : GFXApplicationTemplate<CGLabEmpty>
{
    [STAThread] static void Main() { RunApplication(); }


    protected override void OnMainWindowLoad(object sender, EventArgs args)
    {
        base.RenderDevice.MouseMoveWithLeftBtnDown += (s, e) => 
                  FirstPoint += new DVector2(e.MovDeltaX, e.MovDeltaY);

        RenderDevice.HotkeyRegister(Keys.PageUp,   (s, e) => ++Length);
        RenderDevice.HotkeyRegister(Keys.PageDown, (s, e) => --Length);
    }


    [DisplayNumericProperty(50, 1, "Длинна", 1)]
    public abstract int  Length { get; set; }

    [DisplayNumericProperty(new [] {300d, 200d}, 1, "Точка", -1000)]
    public abstract DVector2 FirstPoint { get; set; }

    [DisplayNumericProperty(0, 0.1, "Угол")]
    public double Angle {
        get { return Get<double>(); }
        set { 
            while (value <    0) value += 360;
            while (value >= 360) value -= 360;
            Set<double>(value);
        }
    }


    protected override void OnDeviceUpdate(object s, GDIDeviceUpdateArgs e)
    {
        Angle += 0.0360*e.Delta;

        var SecondPoint = FirstPoint + Length * new DVector2(
            Math.Cos(Angle * Math.PI / 180),
            Math.Sin(Angle * Math.PI / 180)
        );

        e.Surface.DrawLine( Color.Red.ToArgb(),  FirstPoint,  SecondPoint );
    }

}
