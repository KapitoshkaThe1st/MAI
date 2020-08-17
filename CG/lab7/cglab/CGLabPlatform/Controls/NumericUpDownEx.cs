using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CGLabPlatform
{
    [System.ComponentModel.DesignerCategory("")]
    public class NumericUpDownEx : NumericUpDown
    {
        [Bindable(true)]
        public new decimal Value
        {
            get { return base.Value; }
            set { base.Value = Math.Min(Math.Max(Minimum, value), Maximum);
                  if (base.Value == value)
                      return;
                  var binding = DataBindings["Value"];
                  if (binding == null)
                      return;
                  var updmode = binding.ControlUpdateMode;
                  binding.ControlUpdateMode = ControlUpdateMode.Never;
                  binding.WriteValue();
                  binding.ControlUpdateMode = updmode;                    
            }
        }
    }
}
