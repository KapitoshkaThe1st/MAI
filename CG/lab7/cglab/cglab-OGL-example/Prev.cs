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
    [DisplayNumericProperty(20, 1, "Апроксимация", 3, 30)]
    public abstract int ApproxLevel { get; set; }

    [DisplayNumericProperty(new[] { 0d, 0d, 0.5d }, 0.1, "Наклон", -0.5, 0.5)]
    public abstract DVector3 Slope { get; set; }

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

    public enum LightingMode
    {
        [Description("Плоская")] FLT,
        [Description("По Гуро")] GRNG,
        [Description("По Фонгу")] PHNG
    }

    [DisplayEnumListProperty(LightingMode.FLT, "Режим освещения")]
    public LightingMode LightMode
    {
        get { return Get<LightingMode>(); }
        set
        {
            if (!Set<LightingMode>(value)) return;
            // ... - какой-то код, который надо выполнить при изменении значения
        }
    }

    [DisplayNumericProperty(new[] { 0d, 0d, 200d }, 1, "Позиция источиника света")]
    public abstract DVector3 LightPosition { get; set; }

    [DisplayNumericProperty(new[] { 1d, 1d, 1d }, 0.05, "Цвет источника света", 0d, 1d)]
    public abstract DVector3 LightColor { get; set; }

    [DisplayNumericProperty(new[] { 0.5d, 0.5d, 0.5d }, 0.05, "Цвет объекта", 0d, 1d)]
    public abstract DVector3 ObjectColor { get; set; }

    [DisplayNumericProperty(200.0, 1.0, "Il", 0.0, 200.0)]
    public virtual double Il { get; set; }

    [DisplayNumericProperty(24, 1, "Ip", 1, 256)]
    public virtual int Ip { get; set; }

    [DisplayNumericProperty(1.0, 0.01, "Ia", 0.0, 1.0)]
    public virtual double Ia { get; set; }

    [DisplayNumericProperty(0.10, 0.01, "Ka", 0.0, 1.0)]
    public virtual double Ka { get; set; }
    [DisplayNumericProperty(1.0, 0.01, "Ks", 0.0, 3.0)]
    public virtual double Ks { get; set; }
    [DisplayNumericProperty(1.0, 0.01, "Kd", 0.0, 3.0)]
    public virtual double Kd { get; set; }

    [DisplayNumericProperty(5.0, 1.0, "K", 0.0, 100d)]
    public virtual double K { get; set; }

    [DllImport("kernel32.dll")]
    static extern bool AttachConsole(int dwProcessId);
    private const int ATTACH_PARENT_PROCESS = -1;

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool AllocConsole();

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    static extern bool FreeConsole();

    [STAThread]
    static void Main()
    {
        AllocConsole();

        RunApplication();

        FreeConsole();
    }

    public class Vertex
    {
        public DVector4 _Point; // точка в локальной системе координат
        public DVector4 Point; // точка в мировой\видовой сиситеме координат

        public List<Polygon> Polygon;

        public DVector4 _Normal;
        public DVector4 Normal;

        public Vertex(DVector3 point)
        {
            Polygon = new List<Polygon>();
            _Point = new DVector4(point, 1.0);

            _Normal = DVector4.Zero;
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
        public DVector4 Normal;

        public List<Vertex> Vertex;

        public int Color;

        public Polygon()
        {
            Vertex = new List<Vertex>();
        }
        public Polygon(List<Vertex> verts)
        {
            Vertex = verts;
            _Normal = CrossProduct(verts[0]._Point - verts[1]._Point, verts[1]._Point - verts[2]._Point);
            _Normal /= _Normal.GetLength();

            foreach (Vertex v in verts)
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

        foreach (Vertex vert in Vertecis)
        {
            DVector4 norm = DVector4.Zero;
            foreach (Polygon pol in vert.Polygon)
            {
                norm += pol._Normal;
            }
            vert._Normal = norm / vert.Polygon.Count();
        }
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

        for (int i = 1; i < 5; ++i)
        {
            Polygons[i] = new Polygon(new List<Vertex>() { Vertecis[i - 1], Vertecis[i], Vertecis[i + 5], Vertecis[i - 1 + 5] });
        }
        Polygons[0] = new Polygon(new List<Vertex>() { Vertecis[4], Vertecis[0], Vertecis[5], Vertecis[9] });

        List<Vertex> lowerBase = new List<Vertex>(5);
        for (int i = 0; i < 5; ++i)
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

        foreach (Vertex vert in Vertecis)
        {
            DVector4 norm = DVector4.Zero;
            foreach (Polygon pol in vert.Polygon)
            {
                norm += pol._Normal;
            }
            vert._Normal = norm / vert.Polygon.Count();
        }
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
        for (int i = 1; i < 4; ++i)
        {
            center += Vertecis[i]._Point;
        }

        center /= 4.0;

        foreach (Vertex v in Vertecis)
        {
            v._Point -= center;
        }

        Polygons[0] = new Polygon(new List<Vertex>() { Vertecis[0], Vertecis[1], Vertecis[2] });
        Polygons[1] = new Polygon(new List<Vertex>() { Vertecis[0], Vertecis[2], Vertecis[3] });
        Polygons[2] = new Polygon(new List<Vertex>() { Vertecis[1], Vertecis[0], Vertecis[3] });
        Polygons[3] = new Polygon(new List<Vertex>() { Vertecis[2], Vertecis[1], Vertecis[3] });

        foreach (Vertex vert in Vertecis)
        {
            DVector4 norm = DVector4.Zero;
            foreach (Polygon pol in vert.Polygon)
            {
                norm += pol._Normal;
            }
            vert._Normal = norm / vert.Polygon.Count();
        }
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

        foreach (Vertex vert in v)
        {
            DVector4 norm = DVector4.Zero;
            foreach (Polygon pol in vert.Polygon)
            {
                norm += pol._Normal;
            }
            vert._Normal = norm / vert.Polygon.Count();
        }

        Vertecis = v.ToArray();
        Polygons = p.ToArray();
    }

    private void InclinedCylinder()
    {
        double step = 1.0 / ApproxLevel;

        int k = 2 + (ApproxLevel - 1) * 3;


        DVector3 slopeStep = Slope / ApproxLevel;
        double s = -0.5;

        List<Vertex> v = new List<Vertex>(k * ApproxLevel);
        for (int j = 0; j < k; ++j)
        {
            for (int i = 0; i < ApproxLevel; ++i)
            {

                double angle = 2.0 * Math.PI / ApproxLevel * i;
                double x = Math.Cos(angle);
                double y = Math.Sin(angle);

                if (j < ApproxLevel - 1)
                {
                    double scale = step * (j + 1);
                    v.Add(new Vertex(new DVector3(x * scale + slopeStep.X * s, y * scale + slopeStep.Y * s, -0.5)));
                }
                else if (j >= k - ApproxLevel + 1)
                {
                    double scale = step * (k - j);
                    v.Add(new Vertex(new DVector3(x * scale + slopeStep.X * s, y * scale + slopeStep.Y * s, 0.5)));
                }
                else
                {
                    v.Add(new Vertex(new DVector3(x + slopeStep.X * s, y + slopeStep.Y * s, step * (j - ApproxLevel + 1) - 0.5)));
                    s += 1.0 / ApproxLevel;
                }
            }
        }

        //Console.WriteLine($"{k} {ApproxLevel} -> {count}");

        List<Polygon> p = new List<Polygon>();
        for (int j = 0; j < k - 1; ++j)
        {
            for (int i = 1; i < ApproxLevel; ++i)
            {
                p.Add(new Polygon(new List<Vertex>() { v[j * ApproxLevel + i - 1], v[j * ApproxLevel + i], v[j * ApproxLevel + i + ApproxLevel], v[j * ApproxLevel + i - 1 + ApproxLevel] }));
            }
            p.Add(new Polygon(new List<Vertex>() { v[j * ApproxLevel + ApproxLevel - 1], v[j * ApproxLevel + 0], v[j * ApproxLevel + ApproxLevel], v[j * ApproxLevel + ApproxLevel + ApproxLevel - 1] }));
        }

        // p.Add(new Polygon(new List<Vertex>() { v[j * 5 + i - 1], v[j * 5 + i], v[j * 5 + i + 5], v[j * 5 + i - 1 + 5] }));
        // p.Add(new Polygon(new List<Vertex>() { v[j * 5 + 4], v[j * 5 + 0], v[j * 5 + 5], v[j * 5 + 9]

        List<Vertex> list = v.GetRange(0, ApproxLevel);
        list.Reverse();
        p.Add(new Polygon(list));
        p.Add(new Polygon(v.GetRange(v.Count() - ApproxLevel, ApproxLevel)));

        foreach (Vertex vert in v)
        {
            DVector4 norm = DVector4.Zero;
            foreach (Polygon pol in vert.Polygon)
            {
                norm += pol._Normal;
            }
            vert._Normal = norm / vert.Polygon.Count();
        }

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
            //M44 = 1.0
        };

        DMatrix4 ry = new DMatrix4
        {
            M11 = yCos,
            M13 = ySin,
            M22 = 1.0,
            M31 = -ySin,
            M33 = yCos,
            //M44 = 1.0
        };

        DMatrix4 rz = new DMatrix4
        {
            M11 = zCos,
            M12 = -zSin,
            M21 = zSin,
            M22 = zCos,
            M33 = 1.0,
            //M44 = 1.0
        };

        mat *= rx * ry * rz;
    }

    void DrawPolygonFixGaps2(Polygon p, Graphics g, double Width, double Height)
    {
        DVector3 center = PolygonCenter(p);
        center.X += Width * 0.5;
        center.Y += Height * 0.5;

        int count = p.Vertex.Count;
        PointF[] pp = new PointF[count];

        for (int i = 0; i < count; ++i)
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
        if (p == Polygons.Last())
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

    private void DrawPolygonNormal(Polygon p, Graphics g, double Width, double Height)
    {
        DVector3 center = PolygonCenter(p);

        double amp = 40.0 * WindowScale;

        g.DrawLine(Pens.AliceBlue, (float)(center.X + Width * 0.5 + ShiftX), (float)(center.Y + Height * 0.5 - ShiftY),
                    (float)(center.X + p.Normal.X * amp + Width * 0.5 + ShiftX), (float)(center.Y + p.Normal.Y * amp + Height * 0.5 - ShiftY));

    }

    private void DrawVertexNormal(Vertex v, Graphics g, double Width, double Height)
    {
        double amp = 40.0 * WindowScale;

        g.DrawLine(Pens.Firebrick, (float)(v.Point.X + Width * 0.5 + ShiftX), (float)(v.Point.Y + Height * 0.5 - ShiftY),
                    (float)(v.Point.X + v.Normal.X * amp + Width * 0.5 + ShiftX), (float)(v.Point.Y + v.Normal.Y * amp + Height * 0.5 - ShiftY));

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
    private DVector3 PrevSlope = new DVector3(0.0, 0.0, 0.0);


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
        base.RenderDevice.MouseMoveWithLeftBtnDown += (s, e) =>
        {
            ShiftX += e.MovDeltaX;
            ShiftY -= e.MovDeltaY;
        };
        base.RenderDevice.MouseMoveWithRightBtnDown += (s, e) =>
        {
            //Rotation.X -= e.MovDeltaY;
            //Rotation.Y += e.MovDeltaX;
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

    void Model(double x, double y, double z, ref int cr, ref int cg, ref int cb)
    {
        cr = 4 * (int)Math.Abs(x) % 256;
        cg = 4 * (int)Math.Abs(y) % 256;
        cb = 4 * (int)Math.Abs(z) % 256;
    }

    void Model2(DVector3 normal, DVector3 pos, ref int cr, ref int cg, ref int cb)
    {
        normal /= normal.GetLength();
        cr = (int)(normal.Z * 255);
        cg = (int)(normal.Z * 255);
        cb = (int)(normal.Z * 255);

        if (cr < 0)
            cr = 0;
        if (cg < 0)
            cg = 0;
        if (cb < 0)
            cb = 0;

    }

    int Clamp(int val, int min, int max)
    {
        if(val > max)
        {
            return max;
        }
        if(val < min)
        {
            return min;
        }
        return val;
    }
    double Clamp(double val, double min, double max)
    {
        if (val > max)
        {
            return max;
        }
        if (val < min)
        {
            return min;
        }
        return val;
    }
    void PhongModel(DVector3 normal, DVector3 pos, ref int cr, ref int cg, ref int cb)
    {
        // ambient
        double AmbientComponent = Ka * Ia;

        double CameraZPosition = 220.0;

        // diffuse
        double distance = (new DVector3(0.0, 0.0, CameraZPosition) - pos).GetLength();

        DVector3 L = LightPosition - pos;
        L /= L.GetLength();

        normal /= normal.GetLength();

        double LNcos = DVector3.DotProduct(L, normal);
        double DiffuseComponent = (Kd * Il) / (K + distance) * Clamp(LNcos, 0.0, 1.0);

        // specular
        DVector3 R = DVector3.Reflect(-L, normal);
        R /= R.GetLength();

        DVector3 S = new DVector3(0.0, 0.0, CameraZPosition) - pos;
        S /= S.GetLength();

        double SRcos = DVector3.DotProduct(R, S);
        double SpecularComponent = Il * Ks * Math.Pow(Clamp(SRcos, 0.0, 1.0), Ip) / (K + distance);
        double result = AmbientComponent + DiffuseComponent + SpecularComponent;

        //if(result >= 1.0)
        //{
        //    R /= R.GetLength();
        //}

        cr = Clamp((int)(255 * result * LightColor.X * ObjectColor.X), 0, 255);
        cg = Clamp((int)(255 * result * LightColor.Y * ObjectColor.Y), 0, 255);
        cb = Clamp((int)(255 * result * LightColor.Z * ObjectColor.Z), 0, 255);
    }

    (Vertex, Vertex, Vertex) Sort3(Vertex a, Vertex b, Vertex c)
    {
        double y1 = a.Point.Y;
        double y2 = b.Point.Y;
        double y3 = c.Point.Y;

        if (y1 > y2)
        {
            if (y1 > y3)
            {
                if (y2 > y3)
                {
                    return (a, b, c);
                }
                else
                {
                    return (a, c, b);
                }
            }
            else
            {
                return (c, a, b);
            }
        }
        else
        {
            if (y2 > y3)
            {
                if (y1 > y3)
                {
                    return (b, a, c);
                }
                else
                {
                    return (b, c, a);
                }
            }
            else
            {
                return (c, b, a);
            }
        }
    }


    int toCompactArgb(int a, int r, int g, int b)
    {
        return (((((a << 8) | r) << 8) | g) << 8) | b;
    }

    double Lerp(double f1, double x, double x1, double x2, double f2)
    {
        if (Math.Abs(x - x1) < 0.00001)
        {
            return f1;
        }

        return f1 + (x - x1) * (f2 - f1) / (x2 - x1);
    }

    double Lerp(double f1, double x, double x1, double k)
    {
        if (Math.Abs(x - x1) < 0.00001)
        {
            return f1;
        }

        return f1 + (x - x1) * k;
    }

    void SetPixel(BitmapSurface bs, int x, int y, int cr, int cg, int cb, int Width, int Height)
    {
        int xToDraw = x + Width / 2 + (int)ShiftX;
        int yToDraw = y + Height / 2 - (int)ShiftY;

        if (xToDraw < 0 || xToDraw >= Width || yToDraw < 0 || yToDraw >= Height)
        {
            return;
        }

        unsafe
        {
            int* bitmap = (int*)bs.ImageBits;
            *(bitmap + yToDraw * Width + xToDraw) = toCompactArgb(255, cr, cg, cb);
        }
    }

    void Swap<T>(ref T a, ref T b)
    {
        T temp = a;
        a = b;
        b = temp;
    }

    void FillTriange(Polygon p, Vertex a, Vertex b, Vertex c, BitmapSurface bs, int Width, int Height)
    {
        // points 
        (c, b, a) = Sort3(a, b, c);

        DVector4 pa = a.Point, pb = b.Point, pc = c.Point;
        /*    (A)
              /\
             /  \
            /    \
           /______\
          /      .   (B)
         /   .        
        /.
       (C)
         */

        // normal components
        double M = p.Normal.X;
        double N = p.Normal.Y;
        double K = p.Normal.Z;


        if (Math.Abs(K) <= 0.00001)
        {
            return;
        }

        // line coefs
        double kAC = (pa.X - pc.X) / (pa.Y - pc.Y);
        double kAB = (pa.X - pb.X) / (pa.Y - pb.Y);
        double kCB = (pc.X - pb.X) / (pc.Y - pb.Y);

        // upper part of triangle
        if (Math.Abs(pa.Y - pb.Y) >= 1.0)
        {
            for (int j = (int)pa.Y; j < (int)pb.Y; ++j)
            {
                //int s = (int)(kAC * (j - c.Y) + c.X);
                //int f = (int)(kAB * (j - b.Y) + b.X);
                int s = (int)Lerp(pc.X, j, pc.Y, kAC);
                int f = (int)Lerp(pb.X, j, pb.Y, kAB);

                double snx = Lerp(c.Normal.X, j, pc.Y, pa.Y, a.Normal.X);
                double sny = Lerp(c.Normal.Y, j, pc.Y, pa.Y, a.Normal.Y);
                double snz = Lerp(c.Normal.Z, j, pc.Y, pa.Y, a.Normal.Z);

                double fnx = Lerp(b.Normal.X, j, pb.Y, pa.Y, a.Normal.X);
                double fny = Lerp(b.Normal.Y, j, pb.Y, pa.Y, a.Normal.Y);
                double fnz = Lerp(b.Normal.Z, j, pb.Y, pa.Y, a.Normal.Z);

                if (s > f)
                {
                    Swap<int>(ref s, ref f);
                    Swap<double>(ref snx, ref fnx);
                    Swap<double>(ref sny, ref fny);
                    Swap<double>(ref snz, ref fnz);
                }

                for (int i = s; i <= f; ++i)
                {
                    // z coord of current fragmetnf
                    double z = pa.Z - (M * (i - pa.X) + N * (j - pa.Y)) / K;

                    double nx = Lerp(snx, i, s, f, fnx);
                    double ny = Lerp(sny, i, s, f, fny);
                    double nz = Lerp(snz, i, s, f, fnz);

                    int cr = 0, cg = 0, cb = 0;
                    PhongModel(new DVector3(nx, ny, nz), new DVector3(i, j, z), ref cr, ref cg, ref cb);

                    SetPixel(bs, i, j, cr, cg, cb, Width, Height);
                }
            }
        }

        // lower part of triangle
        if (Math.Abs(pc.Y - pb.Y) >= 1.0)
        {
            for (int j = (int)pb.Y; j <= (int)pc.Y; ++j)
            {
                //int s = (int)(kAC * (j - c.Y) + c.X);
                //int f = (int)(kCB * (j - b.Y) + b.X);
                int s = (int)Lerp(pc.X, j, pc.Y, kAC);
                int f = (int)Lerp(pb.X, j, pb.Y, kCB);

                double snx = Lerp(c.Normal.X, j, pc.Y, pa.Y, a.Normal.X);
                double sny = Lerp(c.Normal.Y, j, pc.Y, pa.Y, a.Normal.Y);
                double snz = Lerp(c.Normal.Z, j, pc.Y, pa.Y, a.Normal.Z);

                double fnx = Lerp(c.Normal.X, j, pc.Y, pb.Y, b.Normal.X);
                double fny = Lerp(c.Normal.Y, j, pc.Y, pb.Y, b.Normal.Y);
                double fnz = Lerp(c.Normal.Z, j, pc.Y, pb.Y, b.Normal.Z);

                if (s > f)
                {
                    Swap<int>(ref s, ref f);
                    Swap<double>(ref snx, ref fnx);
                    Swap<double>(ref sny, ref fny);
                    Swap<double>(ref snz, ref fnz);
                }

                for (int i = s; i <= f; ++i)
                {
                    double z = pa.Z - (M * (i - pa.X) + N * (j - pa.Y)) / K;

                    double nx = Lerp(snx, i, s, f, fnx);
                    double ny = Lerp(sny, i, s, f, fny);
                    double nz = Lerp(snz, i, s, f, fnz);

                    int cr = 0, cg = 0, cb = 0;
                    //PhongModel(p, i, j, z, ref cr, ref cg, ref cb);
                    //Model2(new DVector3(nx, ny, nz), new DVector3(i, j, z), ref cr, ref cg, ref cb);

                    PhongModel(new DVector3(nx, ny, nz), new DVector3(i, j, z), ref cr, ref cg, ref cb);


                    SetPixel(bs, i, j, cr, cg, cb, Width, Height);
                }
            }
        }
    }

    void FillPolygonPhong(Polygon p, BitmapSurface bs, int Width, int Height)
    {
        int count = p.Vertex.Count();
        for (int i = 1; i < count - 1; ++i)
        {
            FillTriange(p, p.Vertex[0], p.Vertex[i], p.Vertex[i + 1], bs, Width, Height);
        }
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

    private void FillPolygonGournag(Polygon p, BitmapSurface bm, double Width, double Height)
    {
        Vertex v1 = p.Vertex[0];

        int cr = 0, cg = 0, cb = 0;

        PhongModel(new DVector3(v1.Normal), new DVector3(v1.Point), ref cr, ref cg, ref cb);
        int color1 = toCompactArgb(255, cr, cg, cb);

        int vertCount = p.Vertex.Count;
        for (int i = 1; i < vertCount - 1; ++i)
        {
            Vertex v2 = p.Vertex[i];
            PhongModel(new DVector3(v2.Normal), new DVector3(v2.Point), ref cr, ref cg, ref cb);
            int color2 = toCompactArgb(255, cr, cg, cb);

            Vertex v3 = p.Vertex[i + 1];
            PhongModel(new DVector3(v3.Normal), new DVector3(v3.Point), ref cr, ref cg, ref cb);
            int color3 = toCompactArgb(255, cr, cg, cb);

            bm.DrawTriangle(color1, v1.Point.X + Width / 2 + ShiftX, v1.Point.Y + Height / 2 - ShiftY,
                                   color2, v2.Point.X + Width / 2 + ShiftX, v2.Point.Y + Height / 2 - ShiftY,
                                   color3, v3.Point.X + Width / 2 + ShiftX, v3.Point.Y + Height / 2 - ShiftY);
        }
    }

    private void FillPolygonFlat(Polygon p, Graphics gr, double Width, double Height)
    {
        DVector3 center = PolygonCenter(p);

        int cr = 0, cg = 0, cb = 0;
        PhongModel(new DVector3(p.Normal), center, ref cr, ref cg, ref cb);

        gr.FillPolygon(new SolidBrush(Color.FromArgb(cr, cg, cb)), p.Vertex.Select(v => new PointF((float)(v.Point.X + Width / 2 + ShiftX), (float)(v.Point.Y + Height / 2 - ShiftY))).ToArray());
    }

    protected override void OnDeviceUpdate(object s, GDIDeviceUpdateArgs e)
    {
        //e.Surface.DrawTriangle(); // для Гуро за глаза

        if (PrevApproxLevel != ApproxLevel || PrevSlope != Slope)
        {

            // В случае, если в потоке приложения потребуется выполнить некий код, 
            // с синхронизацией с потоком рендера, то реализуем это следующим образом
            lock (RenderDevice.LockObj)
            {
                //ComputeObject();
                InclinedCylinder();
                //Cube();
                //Prism();
                //Tetrahedron();
                PrevApproxLevel = ApproxLevel;
            }
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

        DMatrix4 mat = DMatrix4.Identity;

        double scale = 100.0;

        RotateMatrix(ref mat, rotX, rotY, rotZ);
        ScaleMatrix(ref mat, scale * Scale.X * WindowScale, scale * Scale.Y * WindowScale, scale * Scale.Z * WindowScale);

        DMatrix4 normMat = DMatrix3.NormalVecTransf(mat);

        foreach (Vertex v in Vertecis)
        {
            v.Point = mat * v._Point;
        }

        Polygons.QuickSort(p => p.Vertex.Average(v => v.Point.Z));


        foreach (Polygon p in Polygons)
        {
            p.Normal = normMat * p._Normal;
            p.Normal /= p.Normal.GetLength();

            foreach (Vertex v in p.Vertex)
            {
                v.Normal = normMat * v._Normal;
                v.Normal /= v.Normal.GetLength();
            }

            if (p.Normal.Z > 0)
            {
                //SurfaceFillPolygon(p, e.Surface, (int)e.Width, (int)e.Heigh);
                //BitmapFillPolygon(p, bm, (int)e.Width, (int)e.Heigh);
                //GraphicsFillPolygon(p, e.Graphics, (int)e.Width, (int)e.Heigh);
                //DrawPolygon(p, e.Graphics, e.Width, e.Heigh);
                //DrawPolygonFixGaps2(p, e.Graphics, e.Width, e.Heigh);
                //DrawPolygonFixGaps1(p, e.Graphics, e.Width, e.Heigh);

                switch (LightMode)
                {
                    case LightingMode.FLT :
                        FillPolygonFlat(p, e.Graphics, e.Width, e.Heigh);
                        break;
                    case LightingMode.GRNG :
                        FillPolygonGournag(p, e.Surface, e.Width, e.Heigh);
                        break;
                    case LightingMode.PHNG :
                        FillPolygonPhong(p, e.Surface, (int)e.Width, (int)e.Heigh);
                        break;
                }

            }
        }

        //unsafe
        //{
        //    int* bitmap = (int*)e.Surface.ImageBits;
        //    //int stride = e.Surface.stride;
        //    int stride = (int)e.Width;
        //    for (int i = 0; i < 40; ++i)
        //    {
        //        for (int j = 0; j < 100; ++j)
        //        {
        //            *(bitmap + i * stride + j) = toCompactArgb(255, 255, 255, 255);

        //        }
        //    }
        //}

        if (EnableNormals)
        {
            foreach (Polygon p in Polygons)
            {
                if (p.Normal.Z > 0)
                {
                    DrawPolygonNormal(p, e.Graphics, e.Width, e.Heigh);
                }
            }
            foreach (Vertex v in Vertecis)
            {
                DrawVertexNormal(v, e.Graphics, e.Width, e.Heigh);
            }
        }
        if (EnableVertexNumbers)
        {
            foreach (Polygon p in Polygons)
            {
                if (p.Normal.Z > 0)
                {
                    DrawVertexNumbers(p, e.Graphics, e.Width, e.Heigh);
                }
            }
        }
        if (EnableWireframe)
        {
            foreach (Polygon p in Polygons)
            {
                if (p.Normal.Z > 0)
                {

                    int count = p.Vertex.Count;
                    PointF[] pp = new PointF[count];

                    for (int i = 0; i < count; ++i)
                    {
                        DVector4 v = p.Vertex[i].Point;
                        double x = v.X + e.Width * 0.5 + ShiftX;
                        double y = v.Y + e.Heigh * 0.5 - ShiftY;

                        pp[i] = new PointF((float)x, (float)y);
                    }

                    for (int i = 0; i < count - 1; ++i)
                    {
                        e.Graphics.DrawLine(Pens.White, pp[i], pp[i + 1]);
                    }
                    e.Graphics.DrawLine(Pens.White, pp[count - 1], pp[0]);
                }
            }
        }


        //int side = 5;
        //e.Graphics.FillEllipse(Brushes.Red, new Rectangle((int)(LightPosition.X - side / 2 + e.Width / 2), (int)(LightPosition.Y - side / 2 + e.Heigh / 2), side, side));

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