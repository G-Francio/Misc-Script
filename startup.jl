using MyAstroUtils, DataFramesMeta, JLD2, Printf, AstroLib, OhMyREPL

atreplinit() do repl
    try
        @eval using OhMyREPL
    catch e
        @warn "error while importing OhMyREPL" e
    end
end

"""
    remove_missing!(df::DataFrame)

Changes `missing` or `nothing` from a DataFrame into numbers or strings.
Column types are automatically updated.

!!! warning "Beware!"

    This is **NOT** a general conversion function, and it is suitable for
    my personal needs, in the context of managing dataframes for the QUBRICS' selection.
    Using this for any other needs **IS** un-safe, and might cause issues with
    the arbitrary conversion I chose.


!!! warning "Beware! #2"
    
    Using `disallowmissing!` is a safer and better alternative when possible.

---

`missing` and `nothing` values are changed following this schema:  

| Value     | Convert to | Result      |
|:--------- |:---------- |:----------- |
| `nothing` | `String`   | `"Nothing"` |
| `nothing` | `Int`      | `-999`      |
| `nothing` | `Float`    | `-999.`     |
| `missing` | `String`   | `" "`       |
| `missing` | `Int`      | `0`         |
| `missing` | `Float`    | `NaN`       |

`String` and `Number` are left unchanged. Conversion from `missing` or `nothing` to a
numbers casts the number to the appropriate type.
"""
function remove_missing!(df::DataFrame)
    for name in names(df)
        col_type = eltype(df[!, name])
        if col_type == Missing
            df[!, name] = [" " for i in 1:length(df[!, name])]
        elseif col_type == Nothing
            df[!, name] = ["-999" for i in 1:length(df[!, name])]
        elseif isa(col_type, Union)
            non_missing = collect(col_type)[2]
            df[!, name] = parse_entry.(df[!, name], non_missing)
        end
    end
end

function Base.collect(t::Union{Type,DataType,Union{}})
    _collect(t, [])
end

function _collect(t::Type, list)
    t <: Union{} ? push!(list, t) : _collect(t.b, push!(list, t.a))
end

function _collect(t::Union{DataType,Core.TypeofBottom}, list)
    push!(list, t)
end

function parse_entry(x::Nothing, type)
    if type <: AbstractString
        return "Nothing"
    elseif type <: Number
        return convert(type, -999)
    else
        return x
    end
end

function parse_entry(x::Missing, type)
    if type <: AbstractString
        return " "
    elseif type <: AbstractFloat
        return convert(type, NaN)
    elseif type <: Integer
        return convert(type, 0)
    end
end

function parse_entry(x::Number, type)
    x
end

function parse_entry(x::AbstractString, type)
    return x
end


"""
    sanitize_string!(df::DataFrame)

Sanitizes string columns in a DataFrame by converting them to `String` type. Used to avoid issues when writing fits files.

# Arguments
- `df::DataFrame`: The DataFrame to be sanitized.

# Description
The `sanitize_string!` function iterates over each column in the DataFrame `df` and checks if the column's element type is a subtype of `AbstractString`. If it is, the column values are converted to `String` type using `convert.(String, df[!, name])`. The conversion is done in place, modifying the original DataFrame.

# Examples
```
julia> df = DataFrame(name = ["Alice", "Bob", "Charlie"], age = [25, 30, 35], score = [98.5, 87.0, 92.3])
3×3 DataFrame
 Row │ name     age    score   
     │ String⍰  Int64  Float64 
─────┼─────────────────────────
   1 │ Alice       25     98.5
   2 │ Bob         30     87.0
   3 │ Charlie     35     92.3

julia> sanitize_string!(df)

julia> df
3×3 DataFrame
 Row │ name     age    score   
     │ String   Int64  Float64 
─────┼─────────────────────────
   1 │ Alice       25     98.5
   2 │ Bob         30     87.0
   3 │ Charlie     35     92.3
```
"""
function sanitize_string!(df::DataFrame)
    for name in names(df)
        col_type = eltype(df[!, name])
        if col_type <: AbstractString
            df[!, name] = convert.(String, df[!, name])
        end
    end
end


"""
    jldt2df(path, key = "data")

Reads a jld2 file containing a `DataFrame`, returns a `DataFrame`.

The jld2 group key can be specified using the `key` arguments; defaults to `"data"`.

# Example
```
julia> df = jld2df(/path/to/jld2/file)
2×2 DataFrame
 Row │ a      b
     │ Int64  Int64
─────┼──────────────
   1 │     1      3
   2 │     2      4

julia> df = jld2df(/path/to/jld2/file, key = "my_key")
2×2 DataFrame
 Row │ a      b
     │ Int64  Int64
─────┼──────────────
   1 │     1      3
   2 │     2      4
   
```
"""
function jldt2df(path::String; key="data")
    out = jldopen(path, "r") do file
        tt = file[key]
        return tt
    end
    return out
end

# Left for compatibility reasons till I find all scripts that use this function
jld2toDF = jldt2df

"""
    choose_mag(f, default, args...)

Returns the first valid magnitudes in a given list.
Checks for validity using the `f` function provided. If no valid
magnitudes are found, returns `default`.

# Example
```julia-repl
julia> choose_mag(isnan, -1, [1, 2, 3])
1

julia> choose_mag(isnan, -1, [NaN, 2, 3])
2

julia> choose_mag(isnan, -1, [NaN, NaN, NaN])
-1
```
"""
function choose_mag(f::Function, default, args...)
    if length(args) > 0 && !f(args[1])
        return args[1]
    elseif length(args) > 0 && f(args[1])
        return choose_mag(f, default, args[2:end]...)
    else
        return default
    end
end


"""
    update!(df1, df2; on, cols = nothing)

Updates df1 using the values from df2. Only updates values that are shared
among the respective `on` column.

By default, all columns from `df2` are used to update the columns of `df1`,
with the exception of the `on` column.
The optional `cols` argument allows to update only a subset of columns.

# Example
```julia-repl
julia> df1 = DataFrame(ID = [1, 2, 3], Name = ["Alice", "Bob", "Charlie"], Age = [12, 13, 11])
3×3 DataFrame
 Row │ ID     Name     Age   
     │ Int64  String   Int64 
─────┼───────────────────────
   1 │     1  Alice       12
   2 │     2  Bob         13
   3 │     3  Charlie     11

julia> df2 = DataFrame(ID = [2, 3], Name = ["Bob", "Christopher"], Age = [23, 21])
3×3 DataFrame
 Row │ ID     Name         Age   
     │ Int64  String       Int64 
─────┼───────────────────────────
   1 │     1  Alice           22
   2 │     2  Bob             23
   3 │     3  Christopher     21

julia> update!(df1, df2; on = :ID); df1[!, [:Name, :Age]]
3×2 DataFrame
 Row │ Name         Age   
     │ String       Int64 
─────┼────────────────────
   1 │ Alice           12
   2 │ Bob             23
   3 │ Christopher     21

julia> update!(df1, df2; on = :ID, cols = :Name); df1[!, [:Name, :Age]]
3×2 DataFrame
 Row │ Name         Age   
     │ String       Int64 
─────┼────────────────────
   1 │ Alice           12
   2 │ Bob             13
   3 │ Christopher     11
```
"""
function update!(df1::DataFrame, df2::DataFrame; on::Union{Symbol,String}, cols::Union{Symbol,String,Vector,Nothing}=nothing)
    # find position of commmon elements among the dataframes based on the `on` argument
    sort!(df1, on)
    sort!(df2, on)
    inds_df1 = findall(x -> !isnothing(x), indexin(df1[!, on], df2[!, on]))
    inds_df2 = findall(x -> !isnothing(x), indexin(df2[!, on], df1[!, on]))

    # If no columns are specified, update all columns in df2
    update_cols = names(df2)
    # remove on element
    deleteat!(update_cols, findall(x -> x == string(on), update_cols))
    if cols isa Symbol || cols isa String
        update_cols = [cols]
    elseif !isnothing(cols)
        update_cols = cols
    end

    # Create a view to update the DataFrame in place
    update_view = @view df1[inds_df1, :]
    for colname in update_cols
        update_view[!, colname] .= df2[inds_df2, colname]
    end
end

"""
Converts right ascension and declination coordinates from decimal degrees to sexagesimal format.

# Arguments
- `ra`: the right ascension in decimal degrees.
- `dec`: the declination in decimal degrees.

# Returns
A tuple containing the right ascension and declination in sexagesimal format, with the 
right ascension formatted as HH:MM:SS.SSS and the declination formatted as +/-DD:MM:SS.SSS.

# Errors
- Throws an `ArgumentError` if either `ra` or `dec` is not a finite number.

# Examples
```
julia> convert_coord(83.6331, 22.0145)
("05:34:31.944", "+22:00:52.200")
```

"""
function convert_coord(ra::T, dec::T) where {T<:AbstractFloat}
    if !(isfinite(ra) && isfinite(dec))
        throw(ArgumentError("Invalid input: ra=$ra, dec=$dec"))
    end

    sgn = dec >= 0 ? "+" : "-"
    adec = abs(dec)

    ras = @sprintf("%02d:%02d:%06.3f", sixty(ra / 15)...)
    decs = sgn * @sprintf("%02d:%02d:%06.3f", sixty(adec)...)

    return ras, decs
end

"""
    generate_RAs_DECs!(df::DataFrame, ra_name="RAd", dec_name="DECd")

Generates two new columns in a DataFrame `df` containing the right ascension and declination values
respectively for each row. The names of these columns default to "RAs" and "DECs" but can be specified
using the `ra_name` and `dec_name` arguments.
If `RAs`, `DECs` columns do not exist in `df`, empty columns with those names are created.

# Arguments:
- `df::DataFrame`: the DataFrame containing the right ascension and declination values
- `ra_name::Union{String,Symbol}` (optional, default="RAd"): the name of the column containing the right ascension values
- `dec_name::Union{String,Symbol}` (optional, default="DECd"): the name of the column containing the declination values

# Example:
```
using DataFrames

df = DataFrame(RAd=[10.5, 20.3, 30.7], DECd=[-20.1, -30.9, -40.5]);
generate_RAs_DECs!(df)
df
3×4 DataFrame
 Row │ RAd      DECd     RAs           DECs          
     │ Float64  Float64  String        String        
─────┼───────────────────────────────────────────────
   1 │    10.5    -20.1  00:42:00.000  -20:06:00.000
   2 │    20.3    -30.9  01:21:12.000  -30:54:00.000
   3 │    30.7    -40.5  02:02:48.000  -40:30:00.000
```
"""
function generate_RAs_DECs!(df::DataFrame, ra_name::Union{String,Symbol}="RAd", dec_name::Union{String,Symbol}="DECd")
    df[!, :RAs] = "RAs" in names(df) ? df[!, :RAs] : fill("", length(df[!, 1]))
    df[!, :DECs] = "DECs" in names(df) ? df[!, :DECs] : fill("", length(df[!, 1]))

    for row in eachrow(df)
        coord = convert_coord(row[ra_name], row[dec_name])
        row.RAs = coord[1]
        row.DECs = coord[2]
    end
end

