using System;
using System.Linq;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;
using System.ComponentModel;
using System.Collections.Generic;
using CGLabPlatform;

using System.Diagnostics;
using System.Runtime.InteropServices;

// Объявляем производный класс с любым именем от GFXApplicationTemplate<T>. Его реализация 
// находиться в сборке CGLabPlatform, следовательно надо добавить в Reference ссылку на  
// эту сборку (также нужны ссылки на сборки: Sysytem, System.Drawing и System.Windows.Forms) 
// и объявить, что используется простанство имен - using CGLabPlatform. И да верно, класс 
// является  абстрактным - дело в колдунстве, что происходит внутри базового класса - по факту
// он динамически объявляет новый класс производный от нашего класса и создает прокси объект.

public abstract class CGLab01 : GFXApplicationTemplate<CGLab01>
{
    // Точка входа приложения - просто передаем управление базовому классу
    // вызывая метод RunApplication(), дабы он сделал всю оставшуюся работу
    // Впринципе, одной этой строчки, в объявленном выше классе, вполне 
    // достаточно чтобы приложение можно было уже запустить

    [DllImport("kernel32.dll")]
    static extern bool AttachConsole(int dwProcessId);
    private const int ATTACH_PARENT_PROCESS = -1;

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool AllocConsole();

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool FreeConsole();

    [STAThread] static void Main() {
        //AttachConsole(ATTACH_PARENT_PROCESS);
        AllocConsole();

        Console.WriteLine("Probe..");

        RunApplication();

        FreeConsole();
    }

    // ---------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------
    // --- Часть 1: Объявление и работа со свойствами
    // ---------------------------------------------------------------------------------

    // Для добавления свойств и связанных с ними элементов управления на панели свойств
    // приложения нужно пометить открытое (public) свойство одним из следующих аттрибутов:
    //  * DisplayNumericProperty - численное значение, типа int, float, double и т.д.
    //  * DisplayCheckerProperty - переключатель для значения типа bool
    //  * DisplayTextBoxProperty - текстовое значение типа string
    //  * DisplayEnumListProperty - меню выбора для значения типа перечисления (Enum)
    //  У всех аттрибутов первый параметр задает начальное значение, затем значение 
    //  типа string - отображаемое название. Для численных значений дополнительно
    //  можно задать величину шага (инкремент) и граничные значение (мин, макс)
    //  Сами свойства можно определить несколькими способами:

    // 1 - Краткая форма: Просто объявляем свойства как абстрактное или виртуальное
    [DisplayNumericProperty(0, 1, "Сдвиг по X")]
    public abstract int ShiftX { get; set; }

    [DisplayNumericProperty(0, 0.1, "Сдвиг по Y", -1000)]
    public virtual double ShiftY { get; set; }

    [DisplayNumericProperty(new [] {0d, 0d}, 1, "Смещение")]
    public abstract DVector2 Offset { get; set; }

    [DisplayNumericProperty(new[] { 1d, 1d }, 0.1, "Масштабирование")]
    public abstract DVector2 Scale { get; set; }

    [DisplayNumericProperty(0.0, 0.5, "Поворот", 0.0, 360.0)]
    public abstract double RotationAngle { get; set; }

    [DisplayNumericProperty(50, 2, "Апроксимация", 0)]
    public abstract int ApproxLevel { get; set; }

    [DisplayNumericProperty(100.0, 0.1, "Амплитуда", 0.0)]
    public abstract double Amplitude { get; set; }

    // Все остальные формы объявления имеют лишь смысл, когда нужно добавить некий  код, 
    // выполняемый непосредственно при изменении или получении того или иного значения.

    // 2 - Полная форма: Делаем почти все самостоятельно
    //[DisplayCheckerProperty(true, "Включить вращения")]
    //public bool EnableRot {         // Cвойства на деле синтаксический сахар после компиляции
    //    get {                       // посути получаем методы: getSomething() и setSomething()
    //        return _EnableRot;      // Поэтому туда и можно пихать код и поэтому же требуется
    //    }                           // дополнительное поле для хранения связанного значения.
    //    set {
    //        _EnableRot = value;
    //        // ... - какой-то код, который надо выполнить при изменении значения
    //        base.OnPropertyChanged();   // Реализация привязки свойства в обе стороны, без
    //    }                               // этого изменение данного свойства не будет приво-
    //}                                   // дить к обновлению содержимого элемента управления
    //private bool _EnableRot;


    // 3 - Упрощенная форма: Объявляем свойство как виртуальное, base.OnPropertyChanged() 
    //                       будет добавленно за нас, за счет чего код сократится
    [DisplayTextBoxProperty("x = A * cos^3(O), y = A * sin^3(O)", "Заголовок")] 
    public virtual string LabelTxt {
        get { return _LabelTxt; }
        set { _LabelTxt = value;
            // ... - какой-то код, который надо выполнить при изменении значения
        }
    }
    private string _LabelTxt;
  

    // 4 - Еще более упрощенный вариант: Дополнительно к предыдущему избавляемся от
    //     объявления поля, вместо этого значение будет храниться в колекции базового
    //     типа, а доступ к нему осуществляться через методы Get<T> и Set<T>. Конечно
    //     при таком подходе скорость доступа существенно падает, но если не обращаться
    //     к нему в циклак во время отрисовки это не критично.
    //[DisplayEnumListProperty(DrawingMode.DIB, "Метод рисования")]
    //public DrawingMode DrawMode {
    //    get { return Get<DrawingMode>(); }
    //    set { if (!Set<DrawingMode>(value)) return;
    //        // ... - какой-то код, который надо выполнить при изменении значения
    //    }
    //}

    // Выше было использованно свойство для собственного перечесления (enum), как не
    // трудно догадаться это его объявление. Аттрибутами Description - задается выводимый
    // текст, когда лениво их можно опустить.
    public enum DrawingMode
    {
        [Description("Graphics")] GFX,
        [Description("Surface")] DIB
    }

    // ---------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------
    // --- Часть 2: Инициализация данных, управления и поведения приложения
    // ---------------------------------------------------------------------------------


    // Если нужна какая-то инициализация данных при запуске приложения, можно реализовать ее
    // в перегрузке данного события, вызываемого единожды перед отображением окна приложения
    protected override void OnMainWindowLoad(object sender, EventArgs args)
    {
        // Созданное приложение имеет два основных элемента управления:
        // base.RenderDevice - левая часть экрана для рисования
        // base.ValueStorage - правая панель для отображения и редактирования свойств

        // Пример изменения внешниго вида элементов управления (необязательный код)
        base.RenderDevice.BufferBackCol = 0xB0;
        base.ValueStorage.Font = new Font("Arial", 12f);
        base.ValueStorage.ForeColor = Color.Firebrick;
        base.ValueStorage.RowHeight = 30;
        base.ValueStorage.BackColor = Color.BlanchedAlmond;
        base.MainWindow.BackColor = Color.DarkGoldenrod;
        base.ValueStorage.RightColWidth = 50;
        base.VSPanelWidth = 300;
        base.VSPanelLeft = true;
        base.MainWindow.Size = new Size(960, 640);
        
        // Реализация управления мышкой с зажатыми левой и правой кнопкой мыши
        base.RenderDevice.MouseMoveWithLeftBtnDown += (s, e) => {
            ShiftX += e.MovDeltaX;
            ShiftY -= e.MovDeltaY;
        };

        base.RenderDevice.MouseMoveWithRightBtnDown += (s, e) => {
            double centerX = Width / 2 + Offset.X + ShiftX;
            double centerY = Height / 2 - Offset.Y - ShiftY;

            double dx = e.MovDeltaX;
            double dy = e.MovDeltaY;

            double startX = e.X - dx;
            double startY = e.Y - dy;

            double curX = e.X;
            double curY = e.Y;

            double centerStartVecX = startX - centerX;
            double centerStartVecY = startY - centerY;

            double centerCurVecX = curX - centerX;
            double centerCurVecY = curY - centerY;

            // invert Oy axis
            centerCurVecY = -centerCurVecY;
            centerStartVecY = -centerStartVecY;
            dy = -dy;

            double centerStartVecMod = Math.Sqrt(centerStartVecX * centerStartVecX + centerStartVecY * centerStartVecY);
            double centerCurVecMod = Math.Sqrt(centerCurVecX * centerCurVecX + centerCurVecY * centerCurVecY);

            double denum = centerStartVecMod * centerCurVecMod;
            double cos;

            if(denum < Double.Epsilon)
            {
                cos = 1.0;
                Console.WriteLine("denum ~= 0");
            }
            else
            {
                cos = (centerStartVecX * centerCurVecX + centerStartVecY * centerCurVecY) / denum;
            }

            if (cos > 1.0)
                cos = 1.0;

            if (cos < -1.0)
                cos = -1.0;

            double deltaAngle = Math.Acos(cos) * 180.0 / Math.PI;

            double ResultAngle = deltaAngle + RotationAngle;

            if (centerStartVecY * dx + (-centerStartVecX) * dy < 0) // dot product of perpendicular center-start vector and delta vector
            {
                ResultAngle = RotationAngle - deltaAngle;
            }
            else
            {
                ResultAngle = RotationAngle + deltaAngle;
            }

            if (ResultAngle > 360.0)
                ResultAngle -= 360.0;

            if (ResultAngle < 0.0)
                ResultAngle += 360.0;

            RotationAngle = ResultAngle;
        };

        // Реализация управления клавиатурой
        RenderDevice.HotkeyRegister(Keys.Up,    (s, e) => ++ShiftY);
        RenderDevice.HotkeyRegister(Keys.Down,  (s, e) => --ShiftY);
        RenderDevice.HotkeyRegister(Keys.Left,  (s, e) => --ShiftX);
        RenderDevice.HotkeyRegister(Keys.Right, (s, e) => ++ShiftX);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Up,    (s, e) => ShiftY +=10);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Down,  (s, e) => ShiftY -=10);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Left,  (s, e) => ShiftX -=10);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Right, (s, e) => ShiftX +=10);

        InitialWidth = base.RenderDevice.Width;
        InitialHeight = base.RenderDevice.Height;

        PrevApproxLevel = ApproxLevel;
        PrevAmplitude = Amplitude;

        // ... расчет каких-то параметров или инициализация ещё чего-то, если нужно
    }


    // ---------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------
    // --- Часть 3: Формирование изображение и его отрисовка на экране
    // ---------------------------------------------------------------------------------


    // При надобности добавляем нужные поля, методы, классы и тд и тп.

    private int InitialWidth;
    private int InitialHeight;

    private int PrevApproxLevel;
    private List<DVector2> points = null;

    private double PrevAmplitude;

    private int Width;
    private int Height;

    private List<DVector2> ComputePoints()
    {
        var res = new List<DVector2>(ApproxLevel);
        for (int i = 0; i < ApproxLevel + 1; ++i)
        {
            double angle = 2 * Math.PI * i / ApproxLevel;
            double x = Amplitude * Math.Pow(Math.Cos(angle), 3.0);
            double y = Amplitude * Math.Pow(Math.Sin(angle), 3.0); // minus to flip Y axis

            res.Add(new DVector2(x, y));
        }

        return res;
    }

    private DVector2 VecRotate(DVector2 v, double angle)
    {
        double cos = Math.Cos(angle);
        double sin = Math.Sin(angle);

        return new DVector2(v.X * cos - v.Y * sin, v.X * sin + v.Y * cos);
    }

    private (double X, double Y) VecRotate(double x, double y, double angle)
    {
        double cos = Math.Cos(angle);
        double sin = Math.Sin(angle);

        double resX = x * cos - y * sin;
        double resY = x * sin + y * cos;

        return (resX, resY);
    }


    // Перегружаем главный метод. По назначению он анологичен методу OnPaint() и предназначен
    // для формирования изображения. Однако в отличии от оного он выполняется паралелльно в
    // другом потоке и вызывается непрерывно. О текущей частоте вызовов можно судить по 
    // счетчику числа кадров в заголовке окна (конечно в режиме отладки скорость падает).
    // Помимо прочего он обеспечивает более высокую скорость рисования и не мерцает.


    void DrawStripes(Graphics gr, DVector2 center, double x1, double y1, double x2, double y2, double angle, double scale, double WindowScale, bool inverted, double invertAngle)
    {
        double startDx = x1 - center.X;
        double startDy = y1 - center.Y;
        double startDist = Math.Sqrt(startDx * startDx + startDy * startDy);

        double endDx = x2 - center.X;
        double endDy = y2 - center.Y;
        double endDist = Math.Sqrt(endDx * endDx + endDy * endDy);

        double step = 10.0;
        double scaledStep = step * scale * WindowScale;

        if(scale > 0.0)
        {
            int start = (int)(startDist / scaledStep) + 1;
            int end = (int)(endDist / scaledStep) + 1;

            double stripeWidth = 10.0 * WindowScale;

            for (int i = -start; i <= end; ++i)
            {
                if (i == 0)
                {
                    continue;
                }

                double leftX, rightX;
                rightX = stripeWidth / 2 * WindowScale;
                leftX = -stripeWidth / 2 * WindowScale;

                double Y = i * scaledStep;

                if (angle > Math.PI)
                {
                    angle -= Math.PI;
                }

                double lx, ly, rx, ry, lbx, lby;
                if (i % 5 == 0)
                {
                    double labelX = rightX + stripeWidth * WindowScale;
                    double fontSize = 10.0 * WindowScale;
                    rightX *= 2;
                    leftX *= 2;
                    (lbx, lby) = VecRotate(labelX, Y - fontSize / 2, angle + 3.0 * Math.PI / 2.0);
                    double label;
                    label = i * step;
                    if (inverted)
                    {
                        label = -label;
                    }

                    if (RotationAngle >= invertAngle && RotationAngle <= 180.0 + invertAngle)
                    {
                        label = -label;
                    }
                    gr.DrawString(label.ToString(), new Font("Arial", (float)fontSize), Brushes.Black, lbx + center.X, lby + center.Y);
                }

                (lx, ly) = VecRotate(leftX, Y, angle + 3.0 * Math.PI / 2.0);
                (rx, ry) = VecRotate(rightX, Y, angle + 3.0 * Math.PI / 2.0);

                gr.DrawLine(Pens.Black, lx + center.X, ly + center.Y, rx + center.X, ry + center.Y);
            }
        }
    }
    void DrawAxis(Graphics gr, Pen pen, double angle, DVector2 center, double scale, double WindowScale, bool invertedStripes, double invertAngle)
    {
        if(Math.Abs(angle) < Double.Epsilon || Math.Abs(angle - Math.PI) < Double.Epsilon)
        {
            gr.DrawLine(pen, 0, center.Y, Width, center.Y);
            DrawStripes(gr, center, 0, center.Y, Width, center.Y, 0, scale, WindowScale, invertedStripes, invertAngle);
            return;
        }
        if (Math.Abs(angle - Math.PI / 2.0) < Double.Epsilon || Math.Abs(angle - 3.0 * Math.PI / 2.0) < Double.Epsilon)
        {
            gr.DrawLine(pen, center.X, 0, center.X, Height);
            DrawStripes(gr, center, center.X, 0, center.X, Height, Math.PI / 2.0, scale, WindowScale, invertedStripes, invertAngle);
            return;
        }

        double k = Math.Tan(angle);
        double b = center.Y - k * center.X;
        double x1 = -b / k;
        double y1 = 0;
        double x2 = 0.0;
        double y2 = b;

        if(x1 < 0.0 || x1 > Width)
        {
            x1 = Width;
            y1 = k * x1 + b;
            if(y1 > Height)
            {
                y1 = Height;
                x1 = (y1 - b) / k;
            }
        }

        if(y2 < 0.0 || y2 > Height)
        {
            y2 = Height;
            x2 = (y2 - b) / k;
            if(x2 > Width)
            {
                x2 = Width;
                y2 = k * x2 + b;
            }
        }

        if (y1 > y2)
        {
            double temp = x1;
            x1 = x2;
            x2 = temp;
            temp = y1;
            y1 = y2;
            y2 = temp;
        }

        gr.DrawLine(pen, x1, y1, x2, y2);
        DrawStripes(gr, center, x1, y1, x2, y2, angle, scale, WindowScale, invertedStripes, invertAngle);
    }

    static void DrawLine(Graphics gr, double x1, double y1, double x2, double y2)
    {
        const int elipseSize = 5;
        gr.FillEllipse(Brushes.Blue, new Rectangle((int)(x1 - elipseSize), (int)(y1 - elipseSize), elipseSize * 2, elipseSize * 2));
        gr.DrawLine(Pens.BlueViolet, x1, y1, x2, y2);
        gr.FillEllipse(Brushes.Red, new Rectangle((int)(x2 - elipseSize), (int)(y2 - elipseSize), elipseSize * 2, elipseSize * 2));

    }

    static int centerCircleSize = 3;

    protected override void OnDeviceUpdate(object s, GDIDeviceUpdateArgs e)
    {
        Width = (int)e.Width;
        Height = (int)e.Heigh;

        double WindowScale;
        if(Height < Width)
        {
            WindowScale = (double)Height / InitialHeight;
        }
        else
        {
            WindowScale = (double)Width / InitialWidth;
        }

        DVector2 pivot = new DVector2(Width / 2, Height / 2);

        Pen graphPen = new Pen(Brushes.DarkBlue, 2.0f);
        Pen xAxisPen = new Pen(Brushes.Red, 2.0f);
        Pen yAxisPen = new Pen(Brushes.Green, 2.0f);

        if (points == null || PrevApproxLevel != ApproxLevel || PrevAmplitude != Amplitude)
        {
            points = ComputePoints();
            PrevApproxLevel = ApproxLevel;
            PrevAmplitude = Amplitude;
            Console.WriteLine("Recomputed!");
        }

        // ----- 
        double RotationAngleRad = RotationAngle / 180.0 * Math.PI;

        DVector2 center = new DVector2(pivot.X + Offset.X + ShiftX, pivot.Y - Offset.Y - ShiftY);

        DrawAxis(e.Graphics, xAxisPen, RotationAngleRad, center, Math.Abs(Scale.X), WindowScale, false, 180.0);
        DrawAxis(e.Graphics, yAxisPen, RotationAngleRad + Math.PI / 2.0, center, Math.Abs(Scale.Y), WindowScale, true, 90.0);

        e.Graphics.FillEllipse(Brushes.Black, new Rectangle((int)(center.X - centerCircleSize), (int)(center.Y - centerCircleSize), centerCircleSize * 2, centerCircleSize * 2));

        DVector2 prevP = new DVector2(pivot);
        bool firstPointComputed = false;

        foreach (var p in points)
        {
            DVector2 scaledP = (new DVector2(p.X, -p.Y)).Multiply(Scale * WindowScale);

            double sin = Math.Sin(RotationAngleRad);
            double cos = Math.Cos(RotationAngleRad);

            DVector2 rotatedP = VecRotate(scaledP, RotationAngleRad);

            DVector2 transformedP = new DVector2(rotatedP.X + Offset.X + ShiftX + pivot.X,
                                                rotatedP.Y + -Offset.Y - ShiftY + pivot.Y);

            if (firstPointComputed)
            {
                e.Graphics.DrawLine(graphPen, prevP.X, prevP.Y, transformedP.X, transformedP.Y);
            }

            prevP = transformedP;

            firstPointComputed = true;
        }

        e.Graphics.DrawString(LabelTxt, new Font("Arial", 15f), Brushes.Black, 10f, 10f);
    }



    // В случае, если в потоке приложения потребуется выполнить некий код, 
    // с синхронизацией с потоком рендера, то реализуем это следующим образом
    //    lock (RenderDevice.LockObj) {
    //        // ... какой-то код
    //    }


}
