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

public abstract class CGLabEmpty : GFXApplicationTemplate<CGLabEmpty>
{
    [DisplayNumericProperty(1, 1, "Апроксимация", 1)]
    public abstract int ApproxLevel { get; set; }

    [DisplayNumericProperty(new[] { 0d, 0d, 0d }, 1, "Поворот")]
    public abstract DVector3 Rotation { get; set; }

    [DisplayNumericProperty(new[] { 1d, 1d, 1d }, 0.1, "Масштаб", 0.0)]
    public abstract DVector3 Scale { get; set; }

    [DisplayNumericProperty(0, 1, "Сдвиг по X")]
    public abstract int ShiftX { get; set; }

    [DisplayNumericProperty(0, 0.1, "Сдвиг по Y", -1000)]
    public virtual double ShiftY { get; set; }

    public enum ProjectionMode
    {
        [Description("Свободная")] FR,
        [Description("Изометрическая")] ISO,
        [Description("Ортографическая-сверху")] ORT_T,
        [Description("Ортографическая-слева")] ORT_L,
        [Description("Ортографическая-спереди")] ORT_F
    }

    [DisplayEnumListProperty(ProjectionMode.FR, "Проекция")]
    public ProjectionMode ProjMode
    {
        get { return Get<ProjectionMode>(); }
        set
        {
            if (!Set<ProjectionMode>(value)) return;
            // ... - какой-то код, который надо выполнить при изменении значения
        }
    }
    [DisplayCheckerProperty(false, "Показывать нормали")]
    public bool EnableNormals
    {         // Cвойства на деле синтаксический сахар после компиляции
        get
        {                       // посути получаем методы: getSomething() и setSomething()
            return _EnableNormals;      // Поэтому туда и можно пихать код и поэтому же требуется
        }                           // дополнительное поле для хранения связанного значения.
        set
        {
            _EnableNormals = value;
            // ... - какой-то код, который надо выполнить при изменении значения
            base.OnPropertyChanged();   // Реализация привязки свойства в обе стороны, без
        }                               // этого изменение данного свойства не будет приво-
    }                                   // дить к обновлению содержимого элемента управления
    private bool _EnableNormals;

    [DisplayCheckerProperty(false, "Показывать номера вершин")]
    public bool EnableVertexNumbers
    {         // Cвойства на деле синтаксический сахар после компиляции
        get
        {                       // посути получаем методы: getSomething() и setSomething()
            return _EnableVertexNumbers;      // Поэтому туда и можно пихать код и поэтому же требуется
        }                           // дополнительное поле для хранения связанного значения.
        set
        {
            _EnableVertexNumbers = value;
            // ... - какой-то код, который надо выполнить при изменении значения
            base.OnPropertyChanged();   // Реализация привязки свойства в обе стороны, без
        }                               // этого изменение данного свойства не будет приво-
    }                                   // дить к обновлению содержимого элемента управления
    private bool _EnableVertexNumbers;

    [DisplayCheckerProperty(false, "Каркасная визуализация")]
    public bool EnableWireframe
    {         // Cвойства на деле синтаксический сахар после компиляции
        get
        {                       // посути получаем методы: getSomething() и setSomething()
            return _EnableWireframe;      // Поэтому туда и можно пихать код и поэтому же требуется
        }                           // дополнительное поле для хранения связанного значения.
        set
        {
            _EnableWireframe = value;
            // ... - какой-то код, который надо выполнить при изменении значения
            base.OnPropertyChanged();   // Реализация привязки свойства в обе стороны, без
        }                               // этого изменение данного свойства не будет приво-
    }                                   // дить к обновлению содержимого элемента управления
    private bool _EnableWireframe;

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
        AllocConsole();

        RunApplication();

        FreeConsole();
    }

    public class Vertex
    {
        public DVector4 _Point; // точка в локальной системе координат
        public DVector4  Point; // точка в мировой\видовой сиситеме координат

        public List<Polygon> Polygon;

        public Vertex(DVector3 point)
        {
            Polygon = new List<Polygon>();
            _Point = new DVector4(point, 1.0);
        }
    }

    private static DVector4 CrossProduct(DVector4 a, DVector4 b)
    {
        double X = a.Y * b.Z - a.Z * b.Y;
        double Y = a.Z * b.X - a.X * b.Z;
        double Z = a.X * b.Y - a.Y * b.X;

        return new DVector4(X, Y, Z, 0.0);
    }
    public class Polygon
    {
        public DVector4 _Normal; 
        public DVector4  Normal;

        public List<Vertex> Vertex;

        public int Color;

        public Polygon() {
            Vertex = new List<Vertex>();
        }
        public Polygon(List<Vertex> verts)
        {
            Vertex = verts;
            _Normal = CrossProduct(verts[0]._Point - verts[1]._Point, verts[1]._Point - verts[2]._Point);
            _Normal /= _Normal.GetLength();

            foreach(Vertex v in verts)
            {
                v.Polygon.Add(this);
            }
        }
    }

    public Vertex[] Vertecis;
    public Polygon[] Polygons;

    protected void Cube()
    {
        Vertecis = new Vertex[8];
        Polygons = new Polygon[6];

        Vertecis[0] = new Vertex(new DVector3(0.5, -0.5, -0.5));
        Vertecis[1] = new Vertex(new DVector3(0.5, 0.5, -0.5));
        Vertecis[2] = new Vertex(new DVector3(-0.5, 0.5, -0.5));
        Vertecis[3] = new Vertex(new DVector3(-0.5, -0.5, -0.5));
        Vertecis[4] = new Vertex(new DVector3(0.5, -0.5, 0.5));
        Vertecis[5] = new Vertex(new DVector3(0.5, 0.5, 0.5));
        Vertecis[6] = new Vertex(new DVector3(-0.5, 0.5, 0.5));
        Vertecis[7] = new Vertex(new DVector3(-0.5, -0.5, 0.5));

        Polygons[0] = new Polygon(new List<Vertex>() { Vertecis[0], Vertecis[1], Vertecis[5], Vertecis[4] });
        Polygons[1] = new Polygon(new List<Vertex>() { Vertecis[1], Vertecis[2], Vertecis[6], Vertecis[5] });
        Polygons[2] = new Polygon(new List<Vertex>() { Vertecis[2], Vertecis[3], Vertecis[7], Vertecis[6] });
        Polygons[3] = new Polygon(new List<Vertex>() { Vertecis[3], Vertecis[0], Vertecis[4], Vertecis[7] });
        Polygons[4] = new Polygon(new List<Vertex>() { Vertecis[3], Vertecis[2], Vertecis[1], Vertecis[0] });
        Polygons[5] = new Polygon(new List<Vertex>() { Vertecis[4], Vertecis[5], Vertecis[6], Vertecis[7] });
    }

    protected void Prism()
    {
        Vertecis = new Vertex[10];
        Polygons = new Polygon[7];

        for (int i = 0; i < 5; ++i)
        {
            double RotStep = 2.0 * Math.PI / 5.0;
            double x = Math.Cos(i * RotStep);
            double y = Math.Sin(i * RotStep);

            Vertecis[i] = new Vertex(new DVector3(x, y, -0.5));
            Vertecis[i + 5] = new Vertex(new DVector3(x, y, 0.5));
        }

        for(int i = 1; i < 5; ++i)
        {
            Polygons[i] = new Polygon(new List<Vertex>() { Vertecis[i - 1], Vertecis[i], Vertecis[i + 5], Vertecis[i - 1 + 5] });
        }
        Polygons[0] = new Polygon(new List<Vertex>() { Vertecis[4], Vertecis[0], Vertecis[5], Vertecis[9] });

        List<Vertex> lowerBase = new List<Vertex>(5);
        for(int i = 0; i < 5; ++i)
        {
            lowerBase.Add(Vertecis[i]);
        }
        lowerBase.Reverse();
        Polygons[5] = new Polygon(lowerBase);

        List<Vertex> upperBase = new List<Vertex>(5);
        for (int i = 0; i < 5; ++i)
        {
            upperBase.Add(Vertecis[i + 5]);
        }
        Console.WriteLine(upperBase.Count());
        Polygons[6] = new Polygon(upperBase);

    }
    protected void Tetrahedron()
    {
        Vertecis = new Vertex[4];
        Polygons = new Polygon[4];

        double sqrt3 = Math.Sqrt(3);
        double sqrt6 = Math.Sqrt(6);

        Vertecis[0] = new Vertex(new DVector3(0.0, 0.0, 0.0));
        Vertecis[1] = new Vertex(new DVector3(0.5, sqrt3 / 2.0, 0.0));
        Vertecis[2] = new Vertex(new DVector3(1.0, 0.0, 0.0));
        Vertecis[3] = new Vertex(new DVector3(0.5, sqrt3 / 6.0, sqrt6 / 3.0));

        DVector4 center = Vertecis[0]._Point;
        for(int i = 1; i < 4; ++i)
        {
            center += Vertecis[i]._Point;
        }

        center /= 4.0;

        foreach(Vertex v in Vertecis)
        {
            v._Point -= center;
        }

        Polygons[0] = new Polygon(new List<Vertex>() { Vertecis[0], Vertecis[1], Vertecis[2]});
        Polygons[1] = new Polygon(new List<Vertex>() { Vertecis[0], Vertecis[2], Vertecis[3]});
        Polygons[2] = new Polygon(new List<Vertex>() { Vertecis[1], Vertecis[0], Vertecis[3]});
        Polygons[3] = new Polygon(new List<Vertex>() { Vertecis[2], Vertecis[1], Vertecis[3]});
    }

    protected void ComputeObject()
    {
        double step = 1.0 / ApproxLevel;

        int k = 2 + (ApproxLevel - 1) * 3;

        List<Vertex> v = new List<Vertex>();
        for (int j = 0; j < k; ++j)
        {
            for (int i = 0; i < 5; ++i)
            {
                double angle = 2.0 * Math.PI / 5.0 * i;
                double x = Math.Cos(angle);
                double y = Math.Sin(angle);

                if (j < ApproxLevel - 1)
                {
                    double scale = step * (j + 1);
                    v.Add(new Vertex(new DVector3(x * scale, y * scale, -0.5)));
                }
                else if (j >= k - ApproxLevel + 1)
                {
                    double scale = step * (k - j);
                    v.Add(new Vertex(new DVector3(x * scale, y * scale, 0.5)));
                }
                else
                {
                    v.Add(new Vertex(new DVector3(x, y, step * (j - ApproxLevel + 1) - 0.5)));
                }
            }
        }

        List<Polygon> p = new List<Polygon>();
        for (int j = 0; j < k - 1; ++j)
        {
            for (int i = 1; i < 5; ++i)
            {
                p.Add(new Polygon(new List<Vertex>() { v[j * 5 + i - 1], v[j * 5 + i], v[j * 5 + i + 5], v[j * 5 + i - 1 + 5] }));
            }
            p.Add(new Polygon(new List<Vertex>() { v[j * 5 + 4], v[j * 5 + 0], v[j * 5 + 5], v[j * 5 + 9] }));
        }

        List<Vertex> list = v.GetRange(0, 5);
        list.Reverse();
        p.Add(new Polygon(list));
        p.Add(new Polygon(v.GetRange(v.Count() - 5, 5)));

        Vertecis = v.ToArray();
        Polygons = p.ToArray();
    }

    double rotX = 0.0;
    double rotY = 0.0;
    double rotZ = 0.0;

    private double ToRadians(double angle)
    {
        return Math.PI * angle / 180.0;
    }

    private void ScaleMatrix(ref DMatrix4 mat, double sX, double sY, double sZ)
    {
        DMatrix4 m = new DMatrix4
        {
            M11 = sX,
            M22 = sY,
            M33 = sZ
        };

        mat *= m;
    }

    private void RotateMatrix(ref DMatrix4 mat, double aX, double aY, double aZ)
    {
        double xRad = ToRadians(aX);
        double yRad = ToRadians(aY);
        double zRad = ToRadians(aZ);

        double xCos = Math.Cos(xRad);
        double xSin = Math.Sin(xRad);
        double yCos = Math.Cos(yRad);
        double ySin = Math.Sin(yRad);
        double zCos = Math.Cos(zRad);
        double zSin = Math.Sin(zRad);

        DMatrix4 rx = new DMatrix4
        {
            M11 = 1.0,
            M22 = xCos,
            M23 = -xSin,
            M32 = xSin,
            M33 = xCos,
            M44 = 1.0
        };

        DMatrix4 ry = new DMatrix4
        {
            M11 = yCos,
            M13 = ySin,
            M22 = 1.0,
            M31 = -ySin,
            M33 = yCos,
            M44 = 1.0
        };

        DMatrix4 rz = new DMatrix4
        {
            M11 = zCos,
            M12 = -zSin,
            M21 = zSin,
            M22 = zCos,
            M33 = 1.0,
            M44 = 1.0
        };

        mat *= rx * ry * rz;
    }

    void DrawPolygonFixGaps2(Polygon p, Graphics g, double Width, double Height)
    {
        DVector3 center = PolygonCenter(p);
        center.X += Width * 0.5 + ShiftX;
        center.Y += Height * 0.5 - ShiftY;

        int count = p.Vertex.Count;
        PointF[] pp = new PointF[count];

        for(int i = 0; i < count; ++i)
        {
            DVector4 v = p.Vertex[i].Point;
            double x = v.X + Width * 0.5 + ShiftX;
            double y = v.Y + Height * 0.5 - ShiftY;

            float allowance = 0.0f;

            if (x > center.X)
            {
                x = Math.Ceiling(x) + allowance;
            }
            if (y > center.Y)
            {
                y = Math.Ceiling(y) + allowance;
            }
            if (x < center.X)
            {
                x = Math.Floor(x) - allowance;
            }
            if (y < center.Y)
            {
                y = Math.Floor(y) - allowance;
            }

            pp[i] = new PointF((float)x, (float)y);
        }

        g.FillPolygon(Brushes.Red, pp);

        if (EnableWireframe)
        {
            for (int i = 0; i < count - 1; ++i)
            {
                g.DrawLine(Pens.White, pp[i], pp[i + 1]);
            }
            g.DrawLine(Pens.White, pp[count - 1], pp[0]);
        }
    }

    void DrawPolygonFixGaps1(Polygon p, Graphics g, double Width, double Height)
    {
        Point[] pp = p.Vertex.Select(v => new Point((int)Math.Round(v.Point.X + Width * 0.5 + ShiftX),
                                                    (int)Math.Round(v.Point.Y + Height * 0.5 - ShiftY))).ToArray();

        g.FillPolygon(Brushes.Red, pp);

        if (EnableWireframe)
        {
            int count = pp.Count();
            Pen pen = new Pen(Color.White, 2.0f);
            for (int i = 0; i < count - 1; ++i)
            {
                g.DrawLine(pen, pp[i], pp[i + 1]);
            }
            g.DrawLine(pen, pp[count - 1], pp[0]);
        }
        else
        {
            int count = pp.Count();
            Pen pen = new Pen(Color.Red, 2.0f);
            for (int i = 0; i < count - 1; ++i)
            {
                g.DrawLine(pen, pp[i], pp[i + 1]);
            }
            g.DrawLine(pen, pp[count - 1], pp[0]);
        }
    }

    void DrawPolygon(Polygon p, Graphics g, double Width, double Height)
    {
        if(p == Polygons.Last())
        {
            Console.WriteLine(p.Vertex.Count());
        }
        Point[] pp = p.Vertex.Select(v => new Point((int)Math.Round(v.Point.X + Width * 0.5 + ShiftX),
                                                    (int)Math.Round(v.Point.Y + Height * 0.5 - ShiftY))).ToArray();

        g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighSpeed;
        g.FillPolygon(Brushes.Red, pp);

        if (EnableWireframe)
        {
            int count = pp.Count();
            for (int i = 0; i < count - 1; ++i)
            {
                g.DrawLine(Pens.White, pp[i], pp[i + 1]);
            }
            g.DrawLine(Pens.White, pp[count - 1], pp[0]);
        }
    }
    void DrawVertecis(Graphics g, double Width, double Height)
    {
        for (int i = 0; i < Vertecis.Length; ++i)
        {
            double x = Vertecis[i].Point.X + Width * 0.5 + ShiftX;
            double y = Vertecis[i].Point.Y + Height * 0.5 - ShiftY;

            double rectSize = 3.0;

            g.DrawEllipse(Pens.Blue, new Rectangle((int)(x - rectSize * 0.5), (int)(y - rectSize * 0.5), (int)rectSize, (int)rectSize));
            g.DrawString($"{i}", new Font("Arial", 10f), Brushes.Black, (float)x, (float)y);
        }
    }
    void DrawVertexNumbers(Polygon p, Graphics g, double Width, double Height)
    {
        foreach (var v in p.Vertex)
        {
            for (int i = 0; i < Vertecis.Length; ++i)
            {
                if (DVector4.ApproxEqual(v.Point, Vertecis[i].Point))
                {
                    g.DrawString(i.ToString(), new Font("Arial", 10f), Brushes.Black, (float)(v.Point.X + Width * 0.5 + ShiftX), (float)(v.Point.Y + Height * 0.5 - ShiftY));
                    break;
                }
            }
        }
    }

    private void DrawNormal(Polygon p, Graphics g, double Width, double Height)
    {
        DVector3 center = PolygonCenter(p);

        double amp = 40.0 * WindowScale;

        g.DrawLine(Pens.AliceBlue, (float)(center.X + Width * 0.5 + ShiftX), (float)(center.Y + Height * 0.5 - ShiftY), 
                    (float)(center.X + p.Normal.X * amp + Width * 0.5 + ShiftX), (float)(center.Y + p.Normal.Y * amp + Height * 0.5 - ShiftY));

    }

    private void DrawAxis(Graphics g, DVector4 ox, DVector4 oy, DVector4 oz, DVector4 pos)
    {
        Pen pr = new Pen(Color.Red, 2.0f);
        Pen pg = new Pen(Color.Green, 2.0f);
        Pen pb = new Pen(Color.Blue, 2.0f);

        // Ox
        g.DrawLine(pr, (float)(pos.X), (float)(pos.Y), (float)(pos.X + ox.X), (float)(pos.Y + ox.Y));
        // Oy
        g.DrawLine(pg, (float)(pos.X), (float)(pos.Y), (float)(pos.X + oy.X), (float)(pos.Y + oy.Y));
        // Oz
        g.DrawLine(pb, (float)(pos.X), (float)(pos.Y), (float)(pos.X + oz.X), (float)(pos.Y + oz.Y));

        g.DrawString("x", new Font("Arial", 10f), Brushes.Red, (float)(pos.X + ox.X + 3), (float)(pos.Y + ox.Y + 3));
        g.DrawString("y", new Font("Arial", 10f), Brushes.Green, (float)(pos.X + oy.X + 3), (float)(pos.Y + oy.Y + 3));
        g.DrawString("z", new Font("Arial", 10f), Brushes.Blue, (float)(pos.X + oz.X + 3), (float)(pos.Y + oz.Y + 3));
    }

    DVector3 PolygonCenter(Polygon p)
    {
        double midX = 0;
        double midY = 0;
        double midZ = 0;
        foreach (Vertex v in p.Vertex)
        {
            midX += v.Point.X;
            midY += v.Point.Y;
            midZ += v.Point.Z;
        }

        int count = p.Vertex.Count();
        midX /= count;
        midY /= count;
        midZ /= count;

        return new DVector3(midX, midY, midZ);
    }

    private int InitialWidth;
    private int InitialHeight;

    private int PrevApproxLevel = 0;

    double WindowScale;

    protected override void OnMainWindowLoad(object sender, EventArgs args)
    {
        // Пример изменения внешниго вида элементов управления (необязательный код)
        base.RenderDevice.BufferBackCol = 0xB0;
        base.ValueStorage.Font = new Font("Arial", 12f);
        //base.ValueStorage.ForeColor = Color.Firebrick;
        base.ValueStorage.RowHeight = 30;
        //base.ValueStorage.BackColor = Color.BlanchedAlmond;
        //base.MainWindow.BackColor = Color.DarkGoldenrod;
        base.ValueStorage.RightColWidth = 50;
        base.VSPanelWidth = 350;
        //base.VSPanelLeft = true;
        base.MainWindow.Size = new Size(960, 640);


        // Реализация управления мышкой с зажатыми левой и правой кнопкой мыши
        base.RenderDevice.MouseMoveWithLeftBtnDown += (s, e) => {
            ShiftX += e.MovDeltaX;
            ShiftY -= e.MovDeltaY;
        };
        base.RenderDevice.MouseMoveWithRightBtnDown += (s, e) => {
            Rotation = new DVector3(Rotation.X - e.MovDeltaY, Rotation.Y + e.MovDeltaX, Rotation.Z);
        };

        // Реализация управления клавиатурой
        RenderDevice.HotkeyRegister(Keys.Up, (s, e) => ++ShiftY);
        RenderDevice.HotkeyRegister(Keys.Down, (s, e) => --ShiftY);
        RenderDevice.HotkeyRegister(Keys.Left, (s, e) => --ShiftX);
        RenderDevice.HotkeyRegister(Keys.Right, (s, e) => ++ShiftX);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Up, (s, e) => ShiftY += 10);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Down, (s, e) => ShiftY -= 10);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Left, (s, e) => ShiftX -= 10);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Right, (s, e) => ShiftX += 10);

        InitialWidth = base.RenderDevice.Width;
        InitialHeight = base.RenderDevice.Height;

        ComputeObject();
        //Cube();
        //Tetrahedron();
        //Prism();
    }

    protected override void OnDeviceUpdate(object s, GDIDeviceUpdateArgs e)
    {
        if(PrevApproxLevel != ApproxLevel)
        {
            ComputeObject();
            PrevApproxLevel = ApproxLevel;
        }

        if (e.Heigh < e.Width)
        {
            WindowScale = (double)e.Heigh / InitialHeight;
        }
        else
        {
            WindowScale = (double)e.Width / InitialWidth;
        }

        switch (ProjMode)
        {
            case ProjectionMode.FR:
                rotX = Rotation.X;
                rotY = Rotation.Y;
                rotZ = Rotation.Z;
                break;
            case ProjectionMode.ISO:
                rotX = -35.0;
                rotY = -45.0;
                rotZ = 0.0;
                break;
            case ProjectionMode.ORT_F:
                rotX = 0.0;
                rotY = 0.0;
                rotZ = 0.0;
                break;
            case ProjectionMode.ORT_L:
                rotX = 0.0;
                rotY = 90.0;
                rotZ = 0.0;
                break;
            case ProjectionMode.ORT_T:
                rotX = -90.0;
                rotY = 0.0;
                rotZ = 0.0;
                break;
        }

        // compute transformation ( scale and rotation ) matrix 
        DMatrix4 mat = DMatrix4.Identity;

        double scale = 100.0;
       
        RotateMatrix(ref mat, rotX, rotY, rotZ);
        ScaleMatrix(ref mat, scale * Scale.X * WindowScale, scale * Scale.Y * WindowScale, scale * Scale.Z * WindowScale);

        // transform verticis of object
        foreach (Vertex v in Vertecis)
        {
            v.Point = mat * v._Point;
        }

        Polygons.QuickSort(p => p.Vertex.Average(v => v.Point.Z));

        // draw main object
        foreach (Polygon p in Polygons)
        {
            p.Normal = CrossProduct(p.Vertex[0].Point - p.Vertex[1].Point, p.Vertex[1].Point - p.Vertex[2].Point);
            p.Normal /= p.Normal.GetLength();

            if (p.Normal.Z > 0)
            {
                DrawPolygonFixGaps2(p, e.Graphics, e.Width, e.Heigh);
            }
        }

        // some optional features
        foreach(Polygon p in Polygons)
        {
            if (p.Normal.Z > 0)
            {
                if (EnableNormals)
                {
                    DrawNormal(p, e.Graphics, e.Width, e.Heigh);
                }
                if (EnableVertexNumbers)
                {
                    DrawVertexNumbers(p, e.Graphics, e.Width, e.Heigh);
                }
            }
        }

        // Drawing axis (in the right lower corner)
        DVector4 ox = mat * (new DVector4(1.0, 0.0, 0.0, 0.0));
        DVector4 oy = mat * (new DVector4(0.0, 1.0, 0.0, 0.0));
        DVector4 oz = mat * (new DVector4(0.0, 0.0, 1.0, 0.0));

        ox = ox / ox.GetLength() * 50.0 * WindowScale;
        oy = oy / oy.GetLength() * 50.0 * WindowScale;
        oz = oz / oz.GetLength() * 50.0 * WindowScale;

        DVector4 pos = new DVector4(e.Width - 70.0 * WindowScale, e.Heigh - 70.0 * WindowScale, 0.0, 0.0);

        DrawAxis(e.Graphics, ox, oy, oz, pos);
    }
}
