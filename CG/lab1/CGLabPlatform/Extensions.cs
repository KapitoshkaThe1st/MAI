using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Collections.Concurrent;
using Ectx = System.Threading.ExecutionContext;
using Sctx = System.Threading.SynchronizationContext;

namespace CGLabPlatform
{
    public static class Extensions
    {
        public delegate T SendAndGetCallback<T>(object state);
        public delegate T SendAndGetCallback_s<T>();
        public delegate void SendOrPostCallback_s();

        public static T Send<T>(this SynchronizationContext sync, SendAndGetCallback<T> callback, object state) {
            T r = default(T);
            sync.Send(s => r = callback.Invoke(s), state);
            return r;
        }

        public static T Send<T>(this SynchronizationContext sync, SendAndGetCallback_s<T> callback) {
            T r = default(T);
            sync.Send(s => r = callback.Invoke(), null);
            return r;
        }

        public static void Send(this SynchronizationContext sync, SendOrPostCallback_s callback) {
            sync.Send(s => callback.Invoke(), null);
        }

        public static void Post(this SynchronizationContext sync, SendOrPostCallback_s callback) {
            sync.Post(s => callback.Invoke(), null);
        }

        // SynchronizationContext для Thread
        public static Sctx GetSyncCtx(this Thread th) {
            return (th == null || th.ExecutionContext == null) ? null : th.ExecutionContext.GetSyncCtx();
        }

        // SynchronizationContext для ExecutionContext
        public static Sctx GetSyncCtx(this Ectx x) {    
            return __get(x);
        }

        static Func<Ectx, Sctx> __get = arg =>
        {
            // нужное значение храниться в скрытом поле (private)
            var fi = typeof(Ectx).GetField("_syncContext", BindingFlags.NonPublic|BindingFlags.Instance);
            var dm = new DynamicMethod("foo", typeof(Sctx), new[] { typeof(Ectx) }, typeof(Ectx), true);
            var il = dm.GetILGenerator();
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldfld, fi);
            il.Emit(OpCodes.Ret);

            // дабы повторно не заниматься кодогенерации, заменяем сами себя полученным делегатом
            __get = (Func<Ectx, Sctx>)dm.CreateDelegate(typeof(Func<Ectx, Sctx>));
            return __get(arg); // возвращение результата первого запроса
        };



        public static void DrawLine(this Graphics graphics, Pen pen, double x1, double y1, double x2, double y2)
        { unchecked {
                graphics.DrawLine(pen,  (int)((x1<0)?(x1-0.5):(x1+0.5)), (int)((y1<0)?(y1-0.5):(y1+0.5)),
                                        (int)((x2<0)?(x2-0.5):(x2+0.5)), (int)((y2<0)?(y2-0.5):(y2+0.5)));
        } }

        public static void DrawString(this Graphics graphics, string s, Font font, Brush brush, double x, double y)
        { unchecked {
            graphics.CompositingMode = CompositingMode.SourceOver;
            graphics.DrawString(s, font, brush, (int)((x<0)?(x-0.5):(x+0.5)), (int)((y<0)?(y-0.5):(y+0.5)));
            if (graphics.SmoothingMode == SmoothingMode.None)
                graphics.CompositingMode = CompositingMode.SourceCopy;
        }}

        public static unsafe void FillRectangle(this BitmapSurface surface, int color, int sx, int sy, int width, int height)
        { unchecked {
            if (!Drawing.ClipRect(surface.Clipper, ref sx, ref sy, ref width, ref height))
                return;
            Drawing.FillRectangle((int*)surface.ImageBits, surface.Width, color, sx, sy, width, height);
        } }

        public static unsafe void DrawLine(this BitmapSurface surface, int color, double x1, double y1, double x2, double y2)
        {
            if (!Drawing.Clip2D(surface.Clipper, ref x1, ref y1, ref x2, ref y2))
                return;
            if (surface.DrawQuality == SurfaceDrawQuality.High) {
                Drawing.DrawLineHQ((int*)surface.ImageBits, surface.Width, color, x1, y1, x2, y2);
            } else {
                Drawing.DrawLineLQ((int*)surface.ImageBits, surface.Width, color, 
                                   (int)(x1+0.5), (int)(y1+0.5), (int)(x2+0.5), (int)(y2+0.5));
            }
        }

        public static unsafe void DrawTriangle(this BitmapSurface surface, int color, double x1, double y1, double x2, double y2, double x3, double y3)
        {
            if (surface.Clipper.PointsInBounds(x1, y1, x2, y2, x3, y3)) {
                Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x1, y1, x2, y2, x3, y3);
                return;
            }

            // NOTE: Отсутсвует реализация частных случаев триангуляции, когда две или три грани полностью отсекаются

            // Соотношение старых и новых координат после отсечения:
            //
            //                 (_x1,_y1)
            //                     O 1 
            //                    / \
            //           (x0,y0) /   \ (x1,y1)    
            //                --+-----+--
            //               \ /       \ /
            //        (x2,y2) +         + (x3,y3)
            //               / \       / \
            //            2 /   \     /   \ 3
            //   (_x2,_y2) O ----+---+---- O (_x3,_y3)
            //             (x4,y4)\ /(x5,y5)        
            int trimed;
            double _x1 = x1, _y1 = y1, _x2 = x2, _y2 = y2, _x3 = x3, _y3 = y3;
            double  x0 = x1,  y0 = y1,  x4 = x2,  y4 = y2,  x5 = x3,  y5 = y3;
            if (!Drawing.Clip2D(surface.Clipper, ref x0, ref y0, ref x2, ref y2)) {
                if (!Drawing.Clip2D(surface.Clipper, ref x1, ref y1, ref x3, ref y3)) {
                    if (!Drawing.Clip2D(surface.Clipper, ref x4, ref y4, ref x5, ref y5)) {
                        return; // все грани не определены
                    } else {
                        return; // грани 1-2 и 1-3 не определены
                    }
                } else if (!Drawing.Clip2D(surface.Clipper, ref x4, ref y4, ref x5, ref y5)) {
                    return; // грани 1-2 и 2-3 не определены
                } else {
                    x0 = x1;  y0 = y1;  x2 = x4;  y2 = y4;
                    trimed = ((x3 == _x3 && y3 == _y3 && x5 == _x3 && y5 == _y3) ? 11 : 15);
                }
            } else if (!Drawing.Clip2D(surface.Clipper, ref x1, ref y1, ref x3, ref y3)) {
                if (!Drawing.Clip2D(surface.Clipper, ref x4, ref y4, ref x5, ref y5)) {
                    return; // грани 1-3 и 2-3 не определены
                } else {
                    x1 = x0;  y1 = y0;  x3 = x5;  y3 = y5;
                    trimed = ((x2 == _x2 && y2 == _y2 && x4 == _x2 && y4 == _y2) ? 13 : 15);
                }
            } else if (!Drawing.Clip2D(surface.Clipper, ref x4, ref y4, ref x5, ref y5)) {
                x4 = x2;  y4 = y2;  x5 = x3;  y5 = y3;
                trimed = ((x0 == _x1 && y0 == _y1 && x1 == _x1 && y1 == _y1) ? 14 : 15);
            } else 
                trimed = ((x0 == _x1 && y0 == _y1 && x1 == _x1 && y1 == _y1) ? 0 : 1)
                       | ((x2 == _x2 && y2 == _y2 && x4 == _x2 && y4 == _y2) ? 0 : 2)
                       | ((x3 == _x3 && y3 == _y3 && x5 == _x3 && y5 == _y3) ? 0 : 4);

            switch (trimed) {
                case 11: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x3, y3, x1, y1, x4, y4);
                         return;
                case 13: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x2, y2, x0, y0, x5, y5);
                         return;
                case 14: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x1, y1, x2, y2, x3, y3);
                         return;
                case  1: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x2, y2, x3, y3, x0, y0);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x3, y3, x0, y0, x1, y1);
                         return;
                case  2: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x1, y1, x3, y3, x4, y4);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x1, y1, x4, y4, x2, y2);
                         return;
                case  4: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x1, y1, x2, y2, x3, y3);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x2, y2, x3, y3, x5, y5);
                         return;
                case  3: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x3, y3, x1, y1, x0, y0);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x3, y3, x0, y0, x2, y2);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x3, y3, x2, y2, x4, y4);
                         return;
                case  5: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x2, y2, x0, y0, x1, y1);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x2, y2, x1, y1, x3, y3);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x2, y2, x3, y3, x5, y5);
                        return;
                case  6: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x1, y1, x2, y2, x4, y4);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x1, y1, x4, y4, x5, y5);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x1, y1, x5, y5, x3, y3);
                         return;
                case 15: 
                case  7: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x0, y0, x1, y1, x3, y3);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x0, y0, x2, y2, x4, y4);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x0, y0, x3, y3, x5, y5);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color, x0, y0, x4, y4, x5, y5);
                         return;
                default: throw new NotImplementedException();
            }
        }

        public static unsafe void DrawTriangle(this BitmapSurface surface, int color1, double x1, double y1, int color2, double x2, double y2, int color3, double x3, double y3)
        {
            if (surface.Clipper.PointsInBounds(x1, y1, x2, y2, x3, y3)) {
                Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color1, x1, y1, color2, x2, y2, color3, x3, y3);
                return;
            }

            // NOTE: Отсутсвует реализация частных случаев триангуляции, когда две или три грани полностью отсекаются

            // Соотношение старых и новых координат после отсечения:
            //
            //                 (_x1,_y1)
            //                     O 1 
            //                    / \
            //           (x0,y0) /   \ (x1,y1)    
            //                --+-----+--
            //               \ /       \ /
            //        (x2,y2) +         + (x3,y3)
            //               / \       / \
            //            2 /   \     /   \ 3
            //   (_x2,_y2) O ----+---+---- O (_x3,_y3)
            //             (x4,y4)\ /(x5,y5)        
            int trimed;
            double _x1 = x1, _y1 = y1, _x2 = x2, _y2 = y2, _x3 = x3, _y3 = y3;
            double  x0 = x1,  y0 = y1,  x4 = x2,  y4 = y2,  x5 = x3,  y5 = y3;
            if (!Drawing.Clip2D(surface.Clipper, ref x0, ref y0, ref x2, ref y2)) {
                if (!Drawing.Clip2D(surface.Clipper, ref x1, ref y1, ref x3, ref y3)) {
                    if (!Drawing.Clip2D(surface.Clipper, ref x4, ref y4, ref x5, ref y5)) {
                        return; // все грани не определены
                    } else {
                        return; // грани 1-2 и 1-3 не определены
                    }
                } else if (!Drawing.Clip2D(surface.Clipper, ref x4, ref y4, ref x5, ref y5)) {
                    return; // грани 1-2 и 2-3 не определены
                } else {
                    x0 = x1;  y0 = y1;  x2 = x4;  y2 = y4;
                    trimed = ((x3 == _x3 && y3 == _y3 && x5 == _x3 && y5 == _y3) ? 11 : 15);
                }
            } else if (!Drawing.Clip2D(surface.Clipper, ref x1, ref y1, ref x3, ref y3)) {
                if (!Drawing.Clip2D(surface.Clipper, ref x4, ref y4, ref x5, ref y5)) {
                    return; // грани 1-3 и 2-3 не определены
                } else {
                    x1 = x0;  y1 = y0;  x3 = x5;  y3 = y5;
                    trimed = ((x2 == _x2 && y2 == _y2 && x4 == _x2 && y4 == _y2) ? 13 : 15);
                }
            } else if (!Drawing.Clip2D(surface.Clipper, ref x4, ref y4, ref x5, ref y5)) {
                x4 = x2;  y4 = y2;  x5 = x3;  y5 = y3;
                trimed = ((x0 == _x1 && y0 == _y1 && x1 == _x1 && y1 == _y1) ? 14 : 15);
            } else 
                trimed = ((x0 == _x1 && y0 == _y1 && x1 == _x1 && y1 == _y1) ? 0 : 1)
                       | ((x2 == _x2 && y2 == _y2 && x4 == _x2 && y4 == _y2) ? 0 : 2)
                       | ((x3 == _x3 && y3 == _y3 && x5 == _x3 && y5 == _y3) ? 0 : 4);

            int c0, c1, c2, c3, c4, c5;
            Drawing.FColor _c1 = color1, _c2 = color2, _c3 = color3;
            if ((trimed & 3) != 0) {
                var cd = (_c2-_c1) / (float)Math.Sqrt((_x2-_x1)*(_x2-_x1) + (_y2-_y1)*(_y2-_y1));
                c0 = _c1 + cd * (float)Math.Sqrt((x0 - _x1)*(x0 - _x1) + (y0 - _y1)*(y0 - _y1));
                c2 = _c1 + cd * (float)Math.Sqrt((x2 - _x1)*(x2 - _x1) + (y2 - _y1)*(y2 - _y1));
            } else { c0 = color1; c2 = color2; }
            if ((trimed & 5) != 0) {
                var cd = (_c3-_c1) / (float)Math.Sqrt((_x3-_x1)*(_x3-_x1) + (_y3-_y1)*(_y3-_y1));
                c1 = _c1 + cd * (float)Math.Sqrt((x1 - _x1)*(x1 - _x1) + (y1 - _y1)*(y1 - _y1));
                c3 = _c1 + cd * (float)Math.Sqrt((x3 - _x1)*(x3 - _x1) + (y3 - _y1)*(y3 - _y1));
            } else { c1 = color1; c3 = color3; }
            if ((trimed & 6) != 0) {
                var cd = (_c3 - _c2) / (float)Math.Sqrt((_x3-_x2)*(_x3-_x2) + (_y3-_y2)*(_y3-_y2));
                c4 = _c2 + cd * (float)Math.Sqrt((x4 - _x2)*(x4 - _x2) + (y4 - _y2)*(y4 - _y2));
                c5 = _c2 + cd * (float)Math.Sqrt((x5 - _x2)*(x5 - _x2) + (y5 - _y2)*(y5 - _y2));
            } else { c4 = color2; c5 = color3; }

            switch (trimed) {
                case 11: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color3, x3, y3, c1, x1, y1, c4, x4, y4);
                         return;
                case 13: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color2, x2, y2, c0, x0, y0, c5, x5, y5);
                         return;
                case 14: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color1, x1, y1, c2, x2, y2, c3, x3, y3);
                         return;
                case  1: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color2, x2, y2, color3, x3, y3, c0, x0, y0);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color3, x3, y3, c0, x0, y0, c1, x1, y1);
                         return;
                case  2: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color1, x1, y1, color3, x3, y3, c4, x4, y4);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color1, x1, y1, c4, x4, y4, c2, x2, y2);
                         return;
                case  4: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color1, x1, y1, color2, x2, y2, c3, x3, y3);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color2, x2, y2, c3, x3, y3, c5, x5, y5);
                         return;
                case  3: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color3, x3, y3, c1, x1, y1, c0, x0, y0);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color3, x3, y3, c0, x0, y0, c2, x2, y2);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color3, x3, y3, c2, x2, y2, c4, x4, y4);
                         return;
                case  5: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color2, x2, y2, c0, x0, y0, c1, x1, y1);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color2, x2, y2, c1, x1, y1, c3, x3, y3);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color2, x2, y2, c3, x3, y3, c5, x5, y5);
                        return;
                case  6: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color1, x1, y1, c2, x2, y2, c4, x4, y4);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color1, x1, y1, c4, x4, y4, c5, x5, y5);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, color1, x1, y1, c5, x5, y5, c3, x3, y3);
                         return;
                case 15: 
                case  7: Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, c0, x0, y0, c1, x1, y1, c3, x3, y3);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, c0, x0, y0, c2, x2, y2, c4, x4, y4);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, c0, x0, y0, c3, x3, y3, c5, x5, y5);
                         Drawing.DrawTriangleLQ((int*)surface.ImageBits, surface.Width, c0, x0, y0, c4, x4, y4, c5, x5, y5);
                         return;
                default: throw new NotImplementedException();
            }
        }


        public static void HotkeyRegister(this Control control, KeyMod mod, Keys key, HandledEventHandler handler)
        {
            Hotkey.Register(control, mod, key, handler);   
        }

        public static void HotkeyRegister(this Control control, Keys key, HandledEventHandler handler)
        {
            Hotkey.Register(control, KeyMod.None, key, handler);
        }
    }

    public static class HelpUtils
    {
        public static void Swap<T>(ref T a, ref T b)
        {
            T t = b;
            b = a;
            a = t;
        }

        public static void Bound<T>(T min, ref T val, T max) where T: IComparable
        {
            if (val.CompareTo(min) < 0)
                val = min;
            if (val.CompareTo(max) > 0)
                val = max;
        }



        public static Image GetImageFromRes(string directory, string filename, string extension, Assembly assembly = null)
        {
            return Bitmap.FromStream(GetStreamFromRes(directory, filename, extension, assembly));
        }

        public static Image GetImageFromRes(string name, Assembly assembly = null)
        {
            return Bitmap.FromStream(GetStreamFromRes(name, assembly));
        }

        public static string GetTextFileFromRes(string directory, string filename, string extension, Assembly assembly = null)
        {
            var stream = GetStreamFromRes(directory, filename, extension, assembly);
            using (TextReader reader = new StreamReader(stream)) {
                return reader.ReadToEnd();
            }
        }

        public static string GetTextFileFromRes(string name, Assembly assembly = null)
        {
            var stream = GetStreamFromRes(name, assembly);
            using (TextReader reader = new StreamReader(stream)) {
                return reader.ReadToEnd();
            }
        }

        public static Stream GetStreamFromRes(string directory, string filename, string extension, Assembly assembly = null)
        {
            var name = new List<string>();
            if (!String.IsNullOrWhiteSpace(directory))
                name.AddRange(directory.Trim('.', '\\', '/').Split('\\', '/'));
            name.Add(filename);
            name.Add(extension.TrimStart('.'));
            var strname = String.Join(".", name);
            return GetStreamFromRes(strname, assembly);
        }

        public static Stream GetStreamFromRes(string name, Assembly assembly = null)
        {
            var resname = GetInternalResourceName(name, ref assembly);
            if (String.IsNullOrEmpty(resname))
                throw new Exception(String.Format("Запрашиваемый файл ресурсов \"{0}\" в сборке \"{1}\" не найден", name, assembly.GetName().Name));
            return assembly.GetManifestResourceStream(resname);
        }

        private static string GetInternalResourceName(string resourceName, ref Assembly resourceAssembly)
        {
            resourceAssembly = resourceAssembly ?? Assembly.GetEntryAssembly();
            var resourcenames = resourceAssembly.GetManifestResourceNames();
            var rootnamespace = String.Join(".", resourcenames
                .Select(n => n.Split('.')).Aggregate((r, n) => {
                    for (int i = 0; i < Math.Min(r.Length, n.Length); ++i)
                        if (r[i] != n[i]) return r.Take(i).ToArray();
                    return r;
                }));
            var searchresname = new List<string>(rootnamespace.Split('.'));
            for (int i = searchresname.Count-1; i >= 0; --i)
                searchresname[i] = String.Join(".", searchresname.Take(i+1));
            searchresname.Add(resourceAssembly.GetName().Name);
            searchresname.Add(resourceName);
            searchresname.Reverse();
            var foundresname = searchresname.Select(n => String.Format("{0}.{1}", n, resourceName)).Intersect(resourcenames).ToList();
            return foundresname.FirstOrDefault();
        }

        public static List<string> GetResourceNames(Assembly assembly = null)
        {
            return (assembly ?? Assembly.GetEntryAssembly()).GetManifestResourceNames().ToList();
        }
    }


    
    public sealed class QueueSynchronizationContext : SynchronizationContext
    {
        private struct ContinuationInformation
        {
            public SendOrPostCallback Continuation;
            public object State;
        }

        private readonly BlockingCollection<ContinuationInformation> queue =
            new BlockingCollection<ContinuationInformation>();

        private readonly int targetThreadId;

        private int recursiveCount = 0;

        public QueueSynchronizationContext() : this(Thread.CurrentThread)
        {
        }

        public QueueSynchronizationContext(Thread owner)
        {
            targetThreadId = owner.ManagedThreadId;
        }

        public override SynchronizationContext CreateCopy()
        {
            return new QueueSynchronizationContext();
        }

        public override void Send(SendOrPostCallback continuation, object state)
        {
            var currentThreadId = Thread.CurrentThread.ManagedThreadId;
            if (currentThreadId == targetThreadId)
            {
                if (recursiveCount < 50)
                {
                    recursiveCount++;

                    continuation(state);

                    recursiveCount--;
                    return;
                }
            }

            var continuationinfo = new ContinuationInformation { Continuation = continuation, State = state };
            queue.Add(continuationinfo);
            while (queue.Contains(continuationinfo))
                Thread.Sleep(1);

        }

        public override void Post(SendOrPostCallback continuation, object state)
        {
            var currentThreadId = Thread.CurrentThread.ManagedThreadId;
            if (currentThreadId == targetThreadId)
            {
                if (recursiveCount < 50)
                {
                    recursiveCount++;

                    continuation(state);

                    recursiveCount--;
                    return;
                }
            }

            queue.Add(new ContinuationInformation { Continuation = continuation, State = state });
        }

        public void Run()
        {
            this.Run(null);
        }

        public void Run(Task task)
        {
            var currentThreadId = Thread.CurrentThread.ManagedThreadId;
            if (currentThreadId != targetThreadId)
            {
                throw new InvalidOperationException();
            }

            if (task != null)
                task.ContinueWith(_ => queue.CompleteAdding());

            if (queue.Count == 0)
                return;
            foreach (var continuationInformation in queue.GetConsumingEnumerable())
            {
                continuationInformation.Continuation(continuationInformation.State);
                if (queue.Count == 0)
                    return;
            }
        }
    }
}
