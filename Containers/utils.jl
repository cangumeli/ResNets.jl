
function auto_convert(w, x, getval)
   type_of_w = typeof(getval(w[1]))
   if  type_of_w != typeof(x)
      return type_of_w(x)
   end
   return x
end
