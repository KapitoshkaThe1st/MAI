using System;
using SharpGL;
using CGLabPlatform;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.IO;
using System.Linq;
using System.Text;

using System.Windows.Forms;

public abstract class CGLabDemoOGL : OGLApplicationTemplate<CGLabDemoOGL>
{
    [STAThread] static void Main() { RunApplication(); }

    #region Свойства

    [DisplayNumericProperty(4, 1, "Размер вершины", 3, 15)]
    public virtual float VertexSize { get; set; }

    [DisplayNumericProperty(1f, 0.05f, "Масштаб", 0.05f)]
    public virtual float Scale
    {
        get { return Get<float>(); }
        set
        {
            if (Set(value))
                RenderDevice.AddScheduleTask((gl, s) =>
                {
                    UpdateProjectionMatrix(gl);
                });
        }
    }
    [DisplayNumericProperty(20, 1, "Аппроксимация", 1)]
    public virtual int ApproxLevel {
        get { return Get<int>(); }
        set
        {
            if (Set(value))
                RenderDevice.AddScheduleTask((gl, s) =>
                {
                    spline.Step = 1.0 / value;
                });
        }
    }

    [DisplayCheckerProperty(false, "Замкнутый сплайн")]
    public virtual bool Loop {
        get
        {
            return Get<bool>();
        }
        set
        {
            if (Set(value))
            {
                spline.Loop = value;
            }
        }
    }
    [DisplayNumericProperty(new[] { 0d, 0d }, 0.01, "Позиция вершины")]
    public virtual DVector2 Pos { get; set; }


[DisplayNumericProperty(0, 0.01, "Сдвиг по X")]
    public virtual double ShiftX
    {
        get { return Get<double>(); }
        set { if (Set(value))
                RenderDevice.AddScheduleTask((gl, s) =>
                {
                    UpdateProjectionMatrix(gl);
                });
        }
    }
    [DisplayNumericProperty(0, 0.01, "Сдвиг по Y")]
    public virtual double ShiftY
    {
        get { return Get<double>(); }
        set { if (Set(value))
                RenderDevice.AddScheduleTask((gl, s) =>
                {
                    UpdateProjectionMatrix(gl);
                });
        }
    }

    #endregion

    private List<Vertex2> vertices = new List<Vertex2>();

    public class Vertex2
    {
        public DVector2 Point { get; set; }
        public Vertex2(double x, double y)
        {
            Point = new DVector2(x, y);
        }
    }

    string text = "";

    public class CatmullRomSpline
    {
        private List<Vertex2> vertices;
        public bool Loop { get; set; }
        public double Step { get; set; }
        public CatmullRomSpline(List<Vertex2> v, bool looped, double Step = 0.01)
        {
            vertices = v;
            Loop = looped;
        }

        public void Draw(OpenGL gl)
        {
            if(vertices.Count < 3)
            {
                return;
            }

            if (Loop)
            {
                gl.Begin(OpenGL.GL_LINE_STRIP);

                int n = vertices.Count;
                if(n > 3)
                {
                    int lim = (int)(n / Step) + 1;

                    for (int i = 1; i <= lim; ++i)
                    {
                        DVector2 v = At((i * Step) % n);
                        gl.Vertex(v.X, v.Y);
                    }
                }
                gl.End();

            }
            else
            {
                int n = vertices.Count;
                int lim = (int)((n - 3) / Step);

                if (n < 4)
                    return;

                gl.Begin(OpenGL.GL_LINE_STRIP);

                for (int i = 0; i <= lim; ++i)
                {
                    DVector2 v = At(1.0 + i * Step);
                    gl.Vertex(v.X, v.Y);
                }

                gl.End();

                gl.Color(1.0, 1.0, 0.0, 1.0);

                DrawArrow(gl, vertices[0].Point, vertices[1].Point);
                DrawArrow(gl, vertices[vertices.Count - 2].Point, vertices[vertices.Count - 1].Point);
            }

            gl.Flush();
        }

        public void Prolong(Vertex2 v)
        {
            vertices.Add(v);
        }

        public DVector2 At(double i)
        {
            if (Loop)
            {
                int n = vertices.Count;

                int k = (int)i;

                int k0 = (k - 1) % n;
                int k1 = (k) % n;
                int k2 = (k + 1) % n;
                int k3 = (k + 2) % n;

                if (k0 < 0)
                    k0 = n + k0;
                if (k1 < 0)
                    k1 = n + k1;
                if (k2 < 0)
                    k2 = n + k2;
                if (k3 < 0)
                    k3 = n + k3;

                DVector2 p0 = vertices[k0].Point;
                DVector2 p1 = vertices[k1].Point;
                DVector2 p2 = vertices[k2].Point;
                DVector2 p3 = vertices[k3].Point;

                double t = i - k;
                double t2 = t * t;
                double t3 = t2 * t;

                return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
            }
            else
            {
                int n = vertices.Count;

                if (i < 1.0 || i > n - 2)
                {
                    return new DVector2(0.0, 0.0);
                }

                int k = (int)i;

                if (k + 2 > n - 1)
                {
                    k--;
                }

                DVector2 p0 = vertices[k - 1].Point;
                DVector2 p1 = vertices[k].Point;
                DVector2 p2 = vertices[k + 1].Point;
                DVector2 p3 = vertices[k + 2].Point;

                double t = i - k;
                double t2 = t * t;
                double t3 = t2 * t;

                return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
            }
        }
    }

    CatmullRomSpline spline;

    DVector2 NormalizeCoords(double x, double y)
    {
        double H = base.RenderDevice.Height;
        double W = base.RenderDevice.Width;

        double normX, normY;
        if (W > H)
        {
            normX = x / H * 2f - W / H;
            normY = -y / H * 2f + 1f;
        }
        else
        {
            normX = x / W * 2f - 1f;
            normY = -y / W * 2f + H / W;
        }

        return new DVector2(normX, normY);
    }

    double NormalizeDistance(double dist)
    {
        DVector2 tmp = NormalizeCoords(0, dist) - NormalizeCoords(0, 0);

        double normDist = Math.Sqrt(tmp.X * tmp.X + tmp.Y * tmp.Y);
        return normDist;
    }

    static double ArrowAngle = Math.PI / 12.0;
    static double ArrowWingsLength = 0.05;
    static void DrawArrow(OpenGL gl, DVector2 from, DVector2 to)
    {
        gl.Begin(OpenGL.GL_LINES);

        gl.Vertex(from.X, from.Y);
        gl.Vertex(to.X, to.Y);

        double tg = (to.X - from.X) / (to.Y - from.Y);
        double angle = Math.Atan(tg);

        if (to.Y < from.Y)
            angle += Math.PI;

        DVector2 lowerWingVec = new DVector2(Math.Sin(angle - ArrowAngle), Math.Cos(angle - ArrowAngle));
        DVector2 upperWingVec = new DVector2(Math.Sin(angle + ArrowAngle), Math.Cos(angle + ArrowAngle));

        // lower wing
        gl.Vertex(to.X, to.Y);
        gl.Vertex(to.X - ArrowWingsLength * lowerWingVec.X, to.Y - ArrowWingsLength * lowerWingVec.Y);

        // upper wing
        gl.Vertex(to.X, to.Y);
        gl.Vertex(to.X - ArrowWingsLength * upperWingVec.X, to.Y - ArrowWingsLength * upperWingVec.Y);

        gl.End();
    }

    Vertex2 ActiveVertex = null;

    DVector2 MousePosition;

    protected override void OnMainWindowLoad(object sender, EventArgs args)
    {
        base.VSPanelWidth = 260;
        base.ValueStorage.RightColWidth = 60;
        base.RenderDevice.VSync = 1;

        #region Обработчики событий мыши и клавиатуры -------------------------------------------------------
        RenderDevice.MouseMoveWithLeftBtnDown += (s, e) => { };
        RenderDevice.MouseMoveWithRightBtnDown += (s, e) => { };
        RenderDevice.MouseLeftBtnDown += (s, e) => {

            DVector2 norm = NormalizeCoords(e.Location.X, e.Location.Y);

            double X = (norm.X - ShiftX) * Scale;
            double Y = (norm.Y - ShiftY) * Scale;

            text = $"{X}  {Y}";
            lock (locker)
            {
                vertices.Add(new Vertex2(X, Y));
            }
        };
        RenderDevice.MouseLeftBtnUp += (s, e) => { };
        RenderDevice.MouseRightBtnDown += (s, e) => {

            DVector2 norm = NormalizeCoords(e.Location.X, e.Location.Y);

            double X = (norm.X - ShiftX) * Scale;
            double Y = (norm.Y - ShiftY) * Scale;

            foreach (Vertex2 v in vertices)
            {
                double dx = v.Point.X - X;
                double dy = v.Point.Y - Y;

                double r = Math.Sqrt(dx * dx + dy * dy);

                double radius = NormalizeDistance(VertexSize) * Scale;

                if (r < radius)
                {
                    ActiveVertex = v;
                    return;
                }
            }
        };

        RenderDevice.MouseRightBtnUp += (s, e) => {
            ActiveVertex = null;
            Pos = new DVector2(0.0, 0.0);
        };
        RenderDevice.MouseMove += (s, e) => {

            MousePosition = new DVector2(e.Location.X + e.MovDeltaX, e.Location.Y + e.MovDeltaY);

            if(ActiveVertex != null)
            {
                double dx = NormalizeDistance(e.MovDeltaX) * Scale;
                double dy = NormalizeDistance(e.MovDeltaY) * Scale;

                if (e.MovDeltaX < 0) dx = -dx;
                if (e.MovDeltaY > 0) dy = -dy;

                double newX = ActiveVertex.Point.X + dx;
                double newY = ActiveVertex.Point.Y + dy;
                ActiveVertex.Point = new DVector2(newX, newY);

                Pos = ActiveVertex.Point;
            }
        };

        // Реализация управления клавиатурой
        RenderDevice.HotkeyRegister(Keys.Up, (s, e) => ShiftY += 0.05);
        RenderDevice.HotkeyRegister(Keys.Down, (s, e) => ShiftY -= 0.05);
        RenderDevice.HotkeyRegister(Keys.Left, (s, e) => ShiftX -= 0.05);
        RenderDevice.HotkeyRegister(Keys.Right, (s, e) => ShiftX += 0.05);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Up, (s, e) => ShiftY += 0.1);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Down, (s, e) => ShiftY -= 0.1);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Left, (s, e) => ShiftX -= 0.1);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Right, (s, e) => ShiftX += 0.1);

        RenderDevice.HotkeyRegister(Keys.C, (s, e) => {
            DVector2 vec = NormalizeCoords(MousePosition.X, MousePosition.Y);

            ShiftX = -vec.X + ShiftX;
            ShiftY = -vec.Y + ShiftY;
        });


        RenderDevice.MouseWheel += (s, e) => {

            float DeltaScale = -e.Delta / 10000.0f;
            Scale += DeltaScale;
        };
        RenderDevice.MouseMoveWithMiddleBtnDown += (s, e) => {
            double dx = NormalizeDistance(e.MovDeltaX);
            double dy = NormalizeDistance(e.MovDeltaY);

            if (e.MovDeltaX < 0) dx = -dx;
            if (e.MovDeltaY > 0) dy = -dy;

            ShiftX += dx;
            ShiftY += dy;
        };

        #endregion

        spline = new CatmullRomSpline(vertices, false, 1.0 / ApproxLevel);

        #region  Инициализация OGL и параметров рендера -----------------------------------------------------
        RenderDevice.AddScheduleTask((gl, s) =>
        {
            gl.FrontFace(OpenGL.GL_CCW);
            gl.Enable(OpenGL.GL_CULL_FACE);
            gl.CullFace(OpenGL.GL_BACK);

            gl.ClearColor(0, 0, 0, 0);

            gl.Enable(OpenGL.GL_DEPTH_TEST);
            gl.DepthFunc(OpenGL.GL_LEQUAL);
            gl.ClearDepth(1.0f);    // 0 - ближе, 1 - далеко 
            gl.ClearStencil(0); 
        });
        #endregion

        #region Обновление матрицы проекции при изменении размеров окна и запуске приложения ----------------
        RenderDevice.Resized += (s, e) =>
        {
            var gl = e.gl;

            UpdateProjectionMatrix(gl);
        };
        #endregion
    }

    private void UpdateProjectionMatrix(OpenGL gl)
    {
        #region Обновление матрицы проекции ---------------------------------------------------------
        double H = gl.RenderContextProvider.Height;
        double W = gl.RenderContextProvider.Width;

        double AspectRatio = W / H;

        gl.MatrixMode(OpenGL.GL_PROJECTION);
        gl.LoadIdentity();


        if (W > H)
        {
            gl.Ortho((-1.0f * AspectRatio - ShiftX) * Scale, (1.0 * AspectRatio - ShiftX) * Scale, (-1.0 - ShiftY) * Scale, (1.0 - ShiftY) * Scale, -100.0, 100.0);
        }
        else
        {
            gl.Ortho((-1.0f - ShiftX) * Scale, (1.0 - ShiftX) * Scale, (-1.0 / AspectRatio - ShiftY) * Scale, (1.0 / AspectRatio - ShiftY) * Scale, -100.0, 100.0);
        }

        #endregion
    }

    object locker = new object();
    
    protected unsafe override void OnDeviceUpdate(object s, OGLDeviceUpdateArgs e)
    {
        var gl = e.gl;

        // Очищаем буфер экрана и буфер глубины (иначе рисоваться все будет поверх старого)
        gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT | OpenGL.GL_STENCIL_BUFFER_BIT);

        gl.Begin(OpenGL.GL_LINE_LOOP);

        gl.Vertex(-1f, -1f);
        gl.Vertex(-1f, 1f);
        gl.Vertex(1f, 1f);
        gl.Vertex(1f, -1f);

        gl.End();

        spline.Draw(gl);

        gl.PointSize(VertexSize);

        lock (locker)
        {
            foreach (Vertex2 v in vertices)
            {
                if (v == ActiveVertex)
                {
                    gl.Color(0.0, 1.0, 0.0, 1.0);
                }
                else if (v == vertices[0] || v == vertices[vertices.Count - 1])
                {
                    gl.Color(0.0, 0.0, 1.0, 1.0);
                }
                else
                {
                    gl.Color(1.0, 0.0, 0.0, 1.0);
                }

                gl.Begin(OpenGL.GL_POINTS);

                gl.Vertex(v.Point.X, v.Point.Y);

                gl.End();
            }
        }

        gl.PointSize(1f);

        gl.Flush();

        gl.Color(1.0, 1.0, 1.0, 1.0);

        return;
    }
}