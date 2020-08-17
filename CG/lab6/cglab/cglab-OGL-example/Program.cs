using System;
using SharpGL;
using CGLabPlatform;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.IO;
using System.Linq;
using System.Text;

using System.ComponentModel;

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
   
    [DisplayNumericProperty(new[] { 0.5d, 0.5d, 0.5d }, 0.05, "Цвет объекта", 0d, 1d)]
    public virtual DVector3 ObjectColor
    {
        get { return Get<DVector3>(); }
        set { if (Set(value)) InclinedCylinder(); }
    }

    [DisplayCheckerProperty(false, "Сетка")]
    public virtual bool Wireframe
    {
        get { return Get<bool>(); }
        set { if (Set(value)) UpdateDisplayMode(); }
    }

    [DisplayCheckerProperty(false, "Оси")]
    public virtual bool Axis { get; set; }

    [DisplayNumericProperty(new[] { 0d, 0d, 3d }, 0.1, "Позиция источиника света")]
    public virtual DVector3 LightPosition { get; set; }

    [DisplayNumericProperty(new[] { 1d, 0d, 0.5d }, 0.1, "Позиция точки")]
    public virtual DVector3 magicPointPosition{ get; set; }

    public enum Mode{
        [Description("До фрагмента")] FRAG,
        [Description("До вершины")] VERT,
    }

    [DisplayEnumListProperty(Mode.VERT, "Режим эффекта")]
    public Mode EffectMode
    {
        get { return Get<Mode>(); }
        set
        {
            if (!Set<Mode>(value)) return;
        }
    }

    [DisplayNumericProperty(0.05f, 0.05f, "Коэффициент для эффекта", 0.0f)]
    public abstract float MagicCoef { get; set; }

    [DisplayNumericProperty(3d, 0.1, "Позиция камеры")]
    public abstract double CameraPosition { get; set; }

    [DisplayNumericProperty(new[] { 6.2, 6.2, 6.2 }, 0.1, "Il", 0.0, 200.0)]
    public virtual DVector3 Il { get; set; }

    [DisplayNumericProperty(50, 1, "Ip", 1, 256)]
    public virtual int Ip { get; set; }

    [DisplayNumericProperty(new[] { 1.0, 1.0, 1.0 }, 0.01, "Ia", 0.0, 1.0)]
    public virtual DVector3 Ia { get; set; }
    [DisplayNumericProperty(new[] { 0.10, 0.10, 0.10 }, 0.01, "Ka", 0.0, 1.0)]
    public virtual DVector3 Ka { get; set; }
    [DisplayNumericProperty(new[] { 0.72, 0.72, 0.720 }, 0.01, "Ks", 0.0, 1.0)]
    public virtual DVector3 Ks { get; set; }
    [DisplayNumericProperty(new[] { 0.5, 0.5, 0.5 }, 0.01, "Kd", 0.0, 1.0)]
    public virtual DVector3 Kd { get; set; }
    [DisplayNumericProperty(1.5, 0.1, "K", 0.0, 100d)]
    public virtual double K { get; set; }

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

    string ReadShader(string path)
    {
        StreamReader reader = new StreamReader(path);
        string text = reader.ReadToEnd();

        return text;
    }
    uint VertexVBOID = 0;
    uint IndexVBOID = 0;

    uint shaderProgram = 0;
    uint vertexShader = 0;
    uint fragmentShader = 0;

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
        };

        #endregion

        InclinedCylinder();

        #region  Инициализация OGL и параметров рендера -----------------------------------------------------
        RenderDevice.AddScheduleTask((gl, s) =>
        {
            gl.FrontFace(OpenGL.GL_CCW);
            gl.Enable(OpenGL.GL_CULL_FACE);
            gl.CullFace(OpenGL.GL_BACK);

            gl.ClearColor(0.2f, 0.6f, 1.0f, 1.0f);


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

            vertexShader = gl.CreateShader(OpenGL.GL_VERTEX_SHADER);
            gl.ShaderSource(vertexShader, ReadShader("VertexShader.vert"));
            gl.CompileShader(vertexShader);

            ShaderLog(gl, vertexShader);

            fragmentShader = gl.CreateShader(OpenGL.GL_FRAGMENT_SHADER);
            gl.ShaderSource(fragmentShader, ReadShader("FragmentShader.frag"));
            gl.CompileShader(fragmentShader);

            ShaderLog(gl, fragmentShader);

            shaderProgram = gl.CreateProgram();

            gl.AttachShader(shaderProgram, vertexShader);
            gl.AttachShader(shaderProgram, fragmentShader);

            gl.LinkProgram(shaderProgram);
            ProgramLog(gl, shaderProgram);

        }, this);
        #endregion

        #region Уничтожение буфера вершин по завершению работы OGL ------------------------------------------
        RenderDevice.Closed += (s, e) => // Событие выполняется в контексте потока OGL при завершении работы
        {
            var gl = e.gl;

            gl.DeleteBuffers(2, new uint[2] { VertexVBOID, IndexVBOID });

            gl.DeleteShader(vertexShader);
            gl.DeleteShader(fragmentShader);
            gl.DeleteProgram(shaderProgram);
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

        #endregion
    }

    private void UpdateModelViewMatrix() // метод вызывается при измении свойств cameraAngle и cameraDistance
    {
        #region Обновление объектно-видовой матрицы ---------------------------------------------------------
        RenderDevice.AddScheduleTask((gl, s) => { });
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

    DMatrix4 Ortho(double left, double right, double top, double bottom, double near, double far)
    {
        return new DMatrix4
        {
            M11 = 2.0 / (right - left),
            M12 = 0.0,
            M13 = 0.0,
            M14 = 0.0,

            M21 = 0.0,
            M22 = -2.0 / (top - bottom),
            M23 = 0.0,
            M24 = 0.0,

            M31 = 0.0,
            M32 = 0.0,
            M33 = 2.0 / (far - near),
            M34 = 0.0,

            M41 = -(right + left) / (right - left),
            M42 = -(top + bottom) / (top - bottom),
            M43 = -(far + near) / (far - near),
            M44 = 1.0
        };
    }

    double ShiftX = 0;
    double ShiftY = 0;

    DMatrix4 OrthoNormalized(double W, double H)
    {
        double AspectRatio = W / H;
        if (W > H)
        {
            return Ortho(-1.0 * AspectRatio - ShiftX, 1.0 * AspectRatio - ShiftX, -1.0 - ShiftY, 1.0 - ShiftY, 2.0, -2.0);
        }
        else
        {
            return Ortho(-1.0 - ShiftX, 1.0 - ShiftX, -1.0 / AspectRatio - ShiftY, 1.0 / AspectRatio - ShiftY, 2.0, -2.0);
        }
    }

    IntPtr BUFFER_OFFSET(uint offset)
    {
        return new IntPtr(offset);
    }

    void ShaderLog(OpenGL gl, uint shader)
    {
        int[] status = new int[1];
        gl.GetShader(shader, OpenGL.GL_COMPILE_STATUS, status);
        if (status[0] == OpenGL.GL_FALSE)
        {
            int[] logLength = new int[1];
            gl.GetShader(shader, OpenGL.GL_INFO_LOG_LENGTH, logLength);

            StringBuilder log = new StringBuilder(logLength[0]);
            gl.GetShaderInfoLog(shader, logLength[0], IntPtr.Zero, log);
            Console.WriteLine(log.ToString());
        }
    }

    void ProgramLog(OpenGL gl, uint program)
    {
        int[] status = new int[1];
        gl.GetProgram(program, OpenGL.GL_LINK_STATUS, status);
        if (status[0] == OpenGL.GL_FALSE)
        {
            int[] logLength = new int[1];

            gl.GetProgram(program, OpenGL.GL_INFO_LOG_LENGTH, logLength);

            StringBuilder log = new StringBuilder(logLength[0]);
            gl.GetProgramInfoLog(program, logLength[0], IntPtr.Zero, log);

            Console.WriteLine(log.ToString());

            gl.DeleteProgram(program);
        }
    }

    private void DrawAxis(OpenGL gl)
    {
        gl.UseProgram(0);
        gl.MatrixMode(OpenGL.GL_MODELVIEW);
        gl.PushMatrix();

        DMatrix4 mat = DMatrix4.Identity;

        RotateMatrix(ref mat, Rotation.X, Rotation.Y, Rotation.Z);
        gl.LoadMatrix(mat.ToArray(true));

        gl.MatrixMode(OpenGL.GL_PROJECTION);
        gl.PushMatrix();

        gl.LoadMatrix(OrthoNormalized(gl.RenderContextProvider.Width, gl.RenderContextProvider.Height).ToArray(true));

        gl.Disable(OpenGL.GL_DEPTH_TEST);

        gl.Begin(OpenGL.GL_LINES);

        const float AxisLength = 0.7f;

        // X-axis
        gl.Color(1f, 0f, 0f);
        gl.Vertex(0f, 0f, 0f);
        gl.Vertex(AxisLength, 0f, 0f);

        // Y-axis
        gl.Color(0f, 1f, 0f);
        gl.Vertex(0f, 0f, 0f);
        gl.Vertex(0f, AxisLength, 0f);

        // Z-axis
        gl.Color(0f, 0f, 1f);
        gl.Vertex(0f, 0f, 0f);
        gl.Vertex(0f, 0f, AxisLength);

        gl.End();

        gl.Enable(OpenGL.GL_DEPTH_TEST);

        gl.MatrixMode(OpenGL.GL_PROJECTION);
        gl.PopMatrix();

        gl.MatrixMode(OpenGL.GL_MODELVIEW);
        gl.PopMatrix();
    }

    void LoadGlobalUniforms(OpenGL gl, uint shaderProgram)
    {
        int KaUniformLocation = gl.GetUniformLocation(shaderProgram, "Ka");
        int KdUniformLocation = gl.GetUniformLocation(shaderProgram, "Kd");
        int KsUniformLocation = gl.GetUniformLocation(shaderProgram, "Ks");
        int IaUniformLocation = gl.GetUniformLocation(shaderProgram, "Ia");
        int IlUniformLocation = gl.GetUniformLocation(shaderProgram, "Il");
        int IpUniformLocation = gl.GetUniformLocation(shaderProgram, "Ip");
        int KUniformLocation = gl.GetUniformLocation(shaderProgram, "K");

        gl.Uniform3(KaUniformLocation, 1, Ka.ToFloatArray());
        gl.Uniform3(KdUniformLocation, 1, Kd.ToFloatArray());
        gl.Uniform3(KsUniformLocation, 1, Ks.ToFloatArray());
        gl.Uniform3(IaUniformLocation, 1, Ia.ToFloatArray());
        gl.Uniform3(IlUniformLocation, 1, Il.ToFloatArray());
        gl.Uniform1(IpUniformLocation, (float)Ip);
        gl.Uniform1(KUniformLocation, (float)K);

        
        int MagicCoefUniformLocation = gl.GetUniformLocation(shaderProgram, "magicCoef");
        int DistToFragUniformLocation = gl.GetUniformLocation(shaderProgram, "distToFrag");

        gl.Uniform1(MagicCoefUniformLocation, MagicCoef);

        if (EffectMode == Mode.FRAG)
        {
            gl.Uniform1(DistToFragUniformLocation, 1);
        }
        else
        {
            gl.Uniform1(DistToFragUniformLocation, 0);
        }

        int CameraPositionUniformLocation = gl.GetUniformLocation(shaderProgram, "CameraPosition");
        int LightPositionUniformLocation = gl.GetUniformLocation(shaderProgram, "LightPosition");
        int magicPointPositionUniformLocation = gl.GetUniformLocation(shaderProgram, "magicPointPosition");

        gl.Uniform1(CameraPositionUniformLocation, (float)CameraPosition);
        gl.Uniform3(LightPositionUniformLocation, 1, LightPosition.ToFloatArray());
        gl.Uniform3(magicPointPositionUniformLocation, 1, magicPointPosition.ToFloatArray());


        int objectColorUniformLocation = gl.GetUniformLocation(shaderProgram, "objectColor");
        gl.Uniform3(objectColorUniformLocation, 1, ObjectColor.ToFloatArray());
    }

    protected unsafe override void OnDeviceUpdate(object s, OGLDeviceUpdateArgs e)
    {
        var gl = e.gl;

        gl.Clear(OpenGL.GL_COLOR_BUFFER_BIT | OpenGL.GL_DEPTH_BUFFER_BIT | OpenGL.GL_STENCIL_BUFFER_BIT);

        gl.UseProgram(shaderProgram);

        DMatrix4 modelMat = DMatrix4.Identity;
        RotateMatrix(ref modelMat, Rotation.X, Rotation.Y, Rotation.Z);
        ScaleMatrix(ref modelMat, Scale.X, Scale.Y, Scale.Z);

        double H = gl.RenderContextProvider.Height;
        double W = gl.RenderContextProvider.Width;

        double AspectRatio = W / H;

        DMatrix4 projectionlMat = OrthoNormalized(W, H);

        DMatrix4 normalMat = DMatrix3.NormalVecTransf(modelMat);

        LoadGlobalUniforms(gl, shaderProgram);

        int modelViewMatrixUniformLocation = gl.GetUniformLocation(shaderProgram, "modelViewMatrix");
        int projectionMatrixUniformLocation = gl.GetUniformLocation(shaderProgram, "projectionMatrix");
        int modelViewNormalMatrixUniformLocation = gl.GetUniformLocation(shaderProgram, "modelViewNormalMatrix");

        gl.UniformMatrix4(modelViewMatrixUniformLocation, 1, false, modelMat.ToArray(true).Select(d => (float)d).ToArray());
        gl.UniformMatrix4(projectionMatrixUniformLocation, 1, false, projectionlMat.ToArray(true).Select(d => (float)d).ToArray());
        gl.UniformMatrix4(modelViewNormalMatrixUniformLocation, 1, false, normalMat.ToArray(true).Select(d => (float)d).ToArray());

        #region Рендинг сцены методом VBO (Vertex Buffer Object) --------------------------------------------

        gl.BindBuffer(OpenGL.GL_ARRAY_BUFFER, VertexVBOID);
        unsafe
        {
            fixed (float* ptr = &GLVertecis[0].vx)
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
        gl.EnableVertexAttribArray(0);    // We like submitting vertices on stream 0 for no special reason
        gl.VertexAttribPointer(0, 3, OpenGL.GL_FLOAT, false, sizeof(GLVertex), BUFFER_OFFSET(0));      // The starting point of the VBO, for the vertices
        gl.EnableVertexAttribArray(1);    // We like submitting normals on stream 1 for no special reason
        gl.VertexAttribPointer(1, 3, OpenGL.GL_FLOAT, false, sizeof(GLVertex), BUFFER_OFFSET(12));     // The starting point of normals, 12 bytes away

        gl.BindBuffer(OpenGL.GL_ELEMENT_ARRAY_BUFFER, IndexVBOID);

        gl.DrawElements(OpenGL.GL_TRIANGLES, indices.Length, OpenGL.GL_UNSIGNED_INT, BUFFER_OFFSET(0));

        gl.DisableVertexAttribArray(0);
        gl.DisableVertexAttribArray(1);

        gl.BindBuffer(OpenGL.GL_ARRAY_BUFFER, 0);
        gl.BindBuffer(OpenGL.GL_ELEMENT_ARRAY_BUFFER, 0);

        #endregion

        if (Axis)
        {
            DrawAxis(gl);
        }
    }

}