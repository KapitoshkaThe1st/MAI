/*  xxx2019   */
#include "mlisp.h"
bool implication_Q(bool x_Q, bool y_Q);
//________________ 
bool implication_Q(bool x_Q, bool y_Q){
  return ((!x_Q) || y_Q);
}

