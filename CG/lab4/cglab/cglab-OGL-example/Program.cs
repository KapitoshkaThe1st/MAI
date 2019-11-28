using System;
using SharpGL;
using CGLabPlatform;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.IO;
using System.Linq;
using System.Text;

// Создание и работа с классом приложения аналогична предыдущим примерам, только в 
// в данном случае наследуемся от шаблона OGLApplicationTemplate<T>, в силу чего
// для вывода графики будет использоваться элемент управления OGLDevice работающий
// через OpenGL (далее OGL). Код OGLDevice размещается в Controls\OGLDevice.cs
public abstract class CGLabDemoOGL : OGLApplicationTemplate<CGLabDemoOGL>
{
    [STAThread] static void Main() { RunApplication(); }

    #region Свойства

    [DisplayNumericProperty(4, 1, "Апроксимация", 3, 30)]
    public virtual int ApproxLevel
    {
        get { return Get<int>(); }
        set { if (Set(value)) InclinedCylinder(); }
    }

    [DisplayNumericProperty(new[] { 0d, 0d }, 0.1, "Наклон", -0.5, 0.5)]
    public virtual DVector2 Slope
    {
        get { return Get<DVector2>(); }
        set { if (Set(value)) InclinedCylinder(); }
    }
    [DisplayNumericProperty(new[] { 0d, 0d, 0d }, 1, "Поворот")]
    public virtual DVector3 Rotation
    {
        get { return Get<DVector3>(); }
        set { if (Set(value)) UpdateModelViewMatrix(); }
    }

    [DisplayNumericProperty(new[] { 1d, 1d, 1d }, 0.1, "Масштаб", 0.0)]
    public virtual DVector3 Scale
    {
        get { return Get<DVector3>(); }
        set { if (Set(value)) UpdateModelViewMatrix(); }
    }
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
    [DisplayNumericProperty(new[] { 0.5d, 0.5d, 0.5d }, 0.05, "Цвет объекта", 0d, 1d)]
    public virtual DVector3 ObjectColor
    {
        get { return Get<DVector3>(); }
        set { if (Set(value)) InclinedCylinder(); }
    }

    [DisplayCheckerProperty(true, "Использовать буфер вершин")]
    public virtual bool useVBO { get; set; }

    [DisplayCheckerProperty(false, "Сетка")]
    public virtual bool Wireframe {
        get { return Get<bool>(); }
        set { if (Set(value)) UpdateDisplayMode(); }
    }

    [DisplayCheckerProperty(true, "Цвет")]
    public virtual bool Color
    {
        get { return Get<bool>(); }
        set { if (Set(value)) UpdateDisplayMode(); }
    }

    [DisplayCheckerProperty(false, "Оси")]
    public virtual bool Axis { get; set; }

    #endregion

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

    private static DVector4 CrossProduct(DVector4 a, DVector4 b)
    {
        double X = a.Y * b.Z - a.Z * b.Y;
        double Y = a.Z * b.X - a.X * b.Z;
        double Z = a.X * b.Y - a.Y * b.X;

        return new DVector4(X, Y, Z, 0.0);
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

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    struct GLVertex
    {
        public readonly float vx, vy, vz;

        public readonly float nx, ny, nz;

        public readonly float r, g, b;
        public GLVertex(float vx, float vy, float vz,
                      float nx, float ny, float nz,
                      float r, float g, float b)
        {
            this.vx = vx;
            this.vy = vy;
            this.vz = vz;
            this.nx = nx;
            this.ny = ny;
            this.nz = nz;
            this.r = r;
            this.g = g;
            this.b = b;
        }
    }

    private GLVertex[] GLVertecis;
    private uint[] indices;

    private void InclinedCylinder()
    {
        double step = 1.0 / ApproxLevel;

        int k = 2 + (ApproxLevel - 1) * 3;

        DVector2 slopeStep = Slope / ApproxLevel;
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

        List<Polygon> p = new List<Polygon>();
        for (int j = 0; j < k - 1; ++j)
        {
            for (int i = 1; i < ApproxLevel; ++i)
            {
                p.Add(new Polygon(new List<Vertex>() { v[j * ApproxLevel + i - 1], v[j * ApproxLevel + i], v[j * ApproxLevel + i + ApproxLevel], v[j * ApproxLevel + i - 1 + ApproxLevel] }));
            }
            p.Add(new Polygon(new List<Vertex>() { v[j * ApproxLevel + ApproxLevel - 1], v[j * ApproxLevel + 0], v[j * ApproxLevel + ApproxLevel], v[j * ApproxLevel + ApproxLevel + ApproxLevel - 1] }));
        }

        List<Vertex> list = v.GetRange(0, ApproxLevel);
        list.Reverse();
        p.Add(new Polygon(list));
        p.Add(new Polygon(v.GetRange(v.Count - ApproxLevel, ApproxLevel)));

        List<uint> ind = new List<uint>();
        List<GLVertex> gv = new List<GLVertex>(v.Capacity);
        foreach (Vertex vert in v)
        {
            DVector4 norm = DVector4.Zero;
            foreach (Polygon pol in vert.Polygon)
            {
                norm += pol._Normal;
            }
            vert._Normal = norm / vert.Polygon.Count;
            gv.Add(new GLVertex((float)vert._Point.X, (float)vert._Point.Y, (float)vert._Point.Z, 
                (float)vert._Normal.X, (float)vert._Normal.Y, (float)vert._Normal.Z,
                (float)ObjectColor.X, (float)ObjectColor.Y, (float)ObjectColor.Z));   
        }

        foreach (Polygon pol in p)
        {
            uint start = (uint)v.FindIndex((Vertex curVert) => curVert == pol.Vertex[0]);
            
            int count = pol.Vertex.Count;
            for (int i = 1; i < count - 1; ++i)
            {
                ind.Add(start);
                ind.Add((uint)v.FindIndex((Vertex curVert) => curVert == pol.Vertex[i]));
                ind.Add((uint)v.FindIndex((Vertex curVert) => curVert == pol.Vertex[i + 1]));
            }
        }
        GLVertecis = gv.ToArray();
        indices = ind.ToArray();
    }

    // Само создание объекта типа OpenGL осуществляется при создании устройства вывода (класс OGLDevice)
    // и доступ к нему можно получить при помощи свойства gl данного объекта (RenderDevice) или объекта
    // типа OGLDeviceUpdateArgs передаваемого в качестве параметра методу OnDeviceUpdate(). Данный метод,
    // как и сама работа с устройством OpenGL реализуются в параллельном потоке. Обращение к устройству
    // OpenGL из другого потока не допускается (создание многопоточного рендера возможно, но это достаточно
    // специфическая архитектура, например рендинг частей экрана в текустуры а потом их объединение).
    // Для большинства функций библиотеки OpenGL при отладке DEBUG конфигурации осуществляется проверка
    // ошибок выполнения и их вывод в окно вывода Microsoft Visual Studio. Поэтому при отладке и написании 
    // кода связанного с OpenGL необходимо также контролировать ошибки библиотеки OpenGL в окне вывода. 

    uint VertexVBOID = 0;
    uint IndexVBOID = 0;
    protected override void OnMainWindowLoad(object sender, EventArgs args)
    {
        base.VSPanelWidth = 260;
        base.ValueStorage.RightColWidth = 60;
        base.RenderDevice.VSync = 1;

        #region Обработчики событий мыши и клавиатуры -------------------------------------------------------
        RenderDevice.MouseMoveWithLeftBtnDown += (s, e) => Rotation += new DVector3(e.MovDeltaY, e.MovDeltaX, 0);

        RenderDevice.MouseMoveWithRightBtnDown += (s, e) => {

            double W = base.RenderDevice.Width;
            double H = base.RenderDevice.Height;

            double AspectRatio = W / H;

            if(W > H)
            {
                ShiftX += ((double)e.MovDeltaX / W * 2) * AspectRatio;
                ShiftY -= (double)e.MovDeltaY / H * 2;
            }
            else
            {
                ShiftX += (double)e.MovDeltaX / W * 2;
                ShiftY -= ((double)e.MovDeltaY / H * 2) / AspectRatio;
            }
        };
        #endregion

        // Как было отмеченно выше вся работа связанная с OGL должна выполнятся в одном потоке. Тут работа с OGL
        // осуществляется в отдельном потоке, а метод OnMainWindowLoad() является событием возбуждаемым потоком
        // пользовательского интерфейса (UI). Поэтой причине весь код ниже добавляется в диспетчер устройства
        // вывода (метод AddScheduleTask() объекта RenderDevice) и выполняется ассинхронно в контексте потока
        // OGL. Сам диспетчер является очередью типа FIFO (First In First Out - т.е. задания обрабатываются 
        // строго в порядке их поступления) и гарантирует, что все задания добавленные в OnMainWindowLoad()
        // будут выполнены до первого вызова метода OnDeviceUpdate() (aka OnPaint)

        InclinedCylinder();

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

        #region Инициализация буфера вершин -----------------------------------------------------------------
        RenderDevice.AddScheduleTask((gl, s) => 
        {
            uint[] VertexVBOIDtempArray = new uint[1];
            gl.GenBuffers(1, VertexVBOIDtempArray);

            VertexVBOID = VertexVBOIDtempArray[0];

            uint[] IndexVBOIDtempArray = new uint[1];
            gl.GenBuffers(1, IndexVBOIDtempArray);

            IndexVBOID = IndexVBOIDtempArray[0];

        }, this);
        #endregion

        #region Уничтожение буфера вершин по завершению работы OGL ------------------------------------------
        RenderDevice.Closed += (s, e) => // Событие выполняется в контексте потока OGL при завершении работы
        {
            var gl = e.gl;

            gl.DeleteBuffers(2, new uint[2] {VertexVBOID, IndexVBOID });
        };
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
            gl.Ortho(-1.0f * AspectRatio - ShiftX, 1.0 * AspectRatio - ShiftX, -1.0 - ShiftY, 1.0 - ShiftY, -100.0, 100.0);
        }
        else
        {
            gl.Ortho(-1.0f - ShiftX, 1.0 - ShiftX, -1.0 / AspectRatio - ShiftY, 1.0 / AspectRatio - ShiftY, -100.0, 100.0);
        }
        #endregion
    }

    private void UpdateModelViewMatrix() // метод вызывается при измении свойств cameraAngle и cameraDistance
    {
        #region Обновление объектно-видовой матрицы ---------------------------------------------------------
        RenderDevice.AddScheduleTask((gl, s) =>
        {
            DMatrix4 modelMat = DMatrix4.Identity;

            RotateMatrix(ref modelMat, Rotation.X, Rotation.Y, Rotation.Z);
            ScaleMatrix(ref modelMat, Scale.X, Scale.Y, Scale.Z);

            gl.MatrixMode(OpenGL.GL_MODELVIEW);
            gl.LoadMatrix(modelMat.ToArray(true));
        });
        #endregion
    }

    private void UpdateDisplayMode() // метод вызывается при измении свойств cameraAngle и cameraDistance
    {
        #region Обновление объектно-видовой матрицы ---------------------------------------------------------
        RenderDevice.AddScheduleTask((gl, s) =>
        {
            if (Wireframe)
            {
                gl.PolygonMode(OpenGL.GL_FRONT_AND_BACK, OpenGL.GL_LINE);
            }
            else
            {
                gl.PolygonMode(OpenGL.GL_FRONT_AND_BACK, OpenGL.GL_FILL);
            }
        });
        #endregion
    }

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
            M33 = sZ,
            M44 = 1.0
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

    IntPtr BUFFER_OFFSET(uint offset)
    {
        return new IntPtr(offset);
    }
    private void DrawAxis(OpenGL gl)
    {
        gl.MatrixMode(OpenGL.GL_MODELVIEW);
        gl.PushMatrix();

        DMatrix4 mat = DMatrix4.Identity;

        RotateMatrix(ref mat, Rotation.X, Rotation.Y, Rotation.Z);
        gl.LoadMatrix(mat.ToArray(true));

        gl.Disable(OpenGL.GL_DEPTH_TEST);

        gl.Begin(OpenGL.GL_LINES);

        const float AxisLength = 0.7f;

        gl.Color(1f, 0f, 0f);
        gl.Vertex(0f, 0f, 0f);
        gl.Vertex(AxisLength, 0f, 0f);

        gl.Color(0f, 1f, 0f);
        gl.Vertex(0f, 0f, 0f);
        gl.Vertex(0f, AxisLength, 0f);

        gl.Color(0f, 0f, 1f);
        gl.Vertex(0f, 0f, 0f);
        gl.Vertex(0f, 0f, AxisLength);

        gl.End();
        gl.Enable(OpenGL.GL_DEPTH_TEST);

        gl.PopMatrix();
    }

    protected unsafe override void OnDeviceUpdate(object s, OGLDeviceUpdateArgs e)
    {
        var gl = e.gl;

        // Очищаем буфер экрана и буфер глубины (иначе рисоваться все будет поверх старого)
        gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT | OpenGL.GL_STENCIL_BUFFER_BIT);

        // Рендинг сцены реализуется одним из двух методов - VB (Vertex Buffer) или VA (Vertex Array), 
        // в зависимости от выбранного пользователем режима.
        if (!useVBO)
        #region Рендинг сцены методом VA (Vertex Array) -----------------------------------------------------
        {
            gl.EnableClientState(OpenGL.GL_VERTEX_ARRAY);
            gl.EnableClientState(OpenGL.GL_NORMAL_ARRAY);
            gl.EnableClientState(OpenGL.GL_COLOR_ARRAY);

            unsafe
            {
                fixed (float* ptr = &GLVertecis[0].vx)
                {
                    float* vptr = ptr + 0;
                    gl.VertexPointer(3, sizeof(GLVertex), vptr);
                    float* nptr = ptr + 3;
                    gl.NormalPointer(sizeof(GLVertex), nptr);
                    float* cptr = ptr + 6;
                    gl.ColorPointer(3, sizeof(GLVertex), cptr);
                }
            }

            if (Wireframe)
            {
                gl.DisableClientState(OpenGL.GL_COLOR_ARRAY);

                gl.PolygonMode(OpenGL.GL_FRONT_AND_BACK, OpenGL.GL_LINE);
                if (Color)
                {
                    gl.Color((ObjectColor.X + 0.5) % 1.0, (ObjectColor.Y + 0.5) % 1.0, (ObjectColor.Z + 0.5) % 1.0);
                    gl.LineWidth(4f);
                }
                else
                {
                    gl.Color(ObjectColor.X, ObjectColor.Y, ObjectColor.Z);
                    gl.LineWidth(2f);
                }

                gl.DrawElements(OpenGL.GL_TRIANGLES, indices.Length, indices);

                gl.EnableClientState(OpenGL.GL_COLOR_ARRAY);
                gl.LineWidth(1f);
            }

            if (Color)
            {
                gl.PolygonMode(OpenGL.GL_FRONT, OpenGL.GL_FILL);
                gl.DrawElements(OpenGL.GL_TRIANGLES, indices.Length, indices);
            }

            gl.DisableClientState(OpenGL.GL_VERTEX_ARRAY);
            gl.DisableClientState(OpenGL.GL_NORMAL_ARRAY);
            gl.DisableClientState(OpenGL.GL_COLOR_ARRAY);
        }
        #endregion
        else
        #region Рендинг сцены методом VBO (Vertex Buffer Object) --------------------------------------------
        {           
            gl.BindBuffer(OpenGL.GL_ARRAY_BUFFER, VertexVBOID);
            unsafe
            {
                fixed(float* ptr = &GLVertecis[0].vx)
                {
                    gl.BufferData(OpenGL.GL_ARRAY_BUFFER, sizeof(GLVertex) * GLVertecis.Length, new IntPtr(ptr), OpenGL.GL_STATIC_DRAW);
                }
            }

            gl.BindBuffer(OpenGL.GL_ELEMENT_ARRAY_BUFFER, IndexVBOID);
            unsafe
            {
                fixed (uint* ptr = &indices[0])
                {
                    gl.BufferData(OpenGL.GL_ELEMENT_ARRAY_BUFFER, sizeof(uint) * indices.Length, new IntPtr(ptr), OpenGL.GL_STATIC_DRAW);
                }
            }

            gl.BindBuffer(OpenGL.GL_ARRAY_BUFFER, VertexVBOID);
            gl.EnableClientState(OpenGL.GL_VERTEX_ARRAY);
            gl.VertexPointer(3, OpenGL.GL_FLOAT, sizeof(GLVertex), BUFFER_OFFSET(0));    // The starting point of the VBO, for the vertices
            gl.EnableClientState(OpenGL.GL_NORMAL_ARRAY);
            gl.NormalPointer(OpenGL.GL_FLOAT, sizeof(GLVertex), BUFFER_OFFSET(12));      // The starting point of normals, 12 bytes away
            gl.EnableClientState(OpenGL.GL_COLOR_ARRAY);
            gl.ColorPointer(3, OpenGL.GL_FLOAT, sizeof(GLVertex), BUFFER_OFFSET(24));      // The starting point of colors, 24 bytes away

            gl.BindBuffer(OpenGL.GL_ELEMENT_ARRAY_BUFFER, IndexVBOID);
           
            if (Wireframe)
            {
                gl.DisableClientState(OpenGL.GL_COLOR_ARRAY);

                gl.PolygonMode(OpenGL.GL_FRONT_AND_BACK, OpenGL.GL_LINE);
                if (Color)
                {
                    gl.Color((ObjectColor.X + 0.5) % 1.0, (ObjectColor.Y + 0.5) % 1.0, (ObjectColor.Z + 0.5) % 1.0);
                    gl.LineWidth(4f);
                }
                else
                {
                    gl.Color(ObjectColor.X, ObjectColor.Y, ObjectColor.Z);
                    gl.LineWidth(2f);
                }

                gl.DrawElements(OpenGL.GL_TRIANGLES, indices.Length, OpenGL.GL_UNSIGNED_INT, BUFFER_OFFSET(0));

                gl.EnableClientState(OpenGL.GL_COLOR_ARRAY);
                gl.LineWidth(1f);
            }

            if (Color)
            {
                gl.PolygonMode(OpenGL.GL_FRONT_AND_BACK, OpenGL.GL_FILL);
                gl.DrawElements(OpenGL.GL_TRIANGLES, indices.Length, OpenGL.GL_UNSIGNED_INT, BUFFER_OFFSET(0));
            }

            gl.DisableClientState(OpenGL.GL_VERTEX_ARRAY);
            gl.DisableClientState(OpenGL.GL_NORMAL_ARRAY);
            gl.DisableClientState(OpenGL.GL_COLOR_ARRAY);

            gl.BindBuffer(OpenGL.GL_ARRAY_BUFFER, 0);
            gl.BindBuffer(OpenGL.GL_ELEMENT_ARRAY_BUFFER, 0);
        }
        #endregion

        if (Axis)
        {
            DrawAxis(gl);
        }
    }
}