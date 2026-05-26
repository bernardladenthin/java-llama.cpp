#!/usr/bin/env python3
"""
JUnit 4 → JUnit 6 (Jupiter) migration script for java-llama.cpp.
"""
import re
import os


def split_args(s):
    """Split comma-separated method arguments respecting nesting and string literals."""
    args = []
    current = []
    depth = 0
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if in_string:
            current.append(c)
            if c == '\\':
                i += 1
                if i < len(s):
                    current.append(s[i])
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
                current.append(c)
            elif c in '([{':
                depth += 1
                current.append(c)
            elif c in ')]}':
                depth -= 1
                current.append(c)
            elif c == ',' and depth == 0:
                args.append(''.join(current))
                current = []
            else:
                current.append(c)
        i += 1
    if current:
        args.append(''.join(current))
    return args


def find_matching_paren(s, start):
    """Find the index of closing paren matching s[start]."""
    depth = 0
    in_string = False
    i = start
    while i < len(s):
        c = s[i]
        if in_string:
            if c == '\\':
                i += 1
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def find_method_body(content, header_end):
    """
    Starting after a method header, find the opening { and return
    (open_brace_index, close_brace_index) for the method body.
    """
    i = header_end
    while i < len(content) and content[i] != '{':
        i += 1
    if i >= len(content):
        return -1, -1
    open_brace = i
    depth = 0
    in_string = False
    while i < len(content):
        c = content[i]
        if in_string:
            if c == '\\':
                i += 1
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return open_brace, i
        i += 1
    return open_brace, -1


def transform_assert_calls(content):
    """
    Transform Assert method calls:
    - assertTrue/assertFalse/assertNotNull/assertNull(msg, x) → (x, msg)  [swap 2-arg]
    - assertEquals/assertArrayEquals/assertSame(msg, exp, act) → (exp, act, msg)  [move 1st to last in 3-arg]

    JUnit 4: message is first arg in multi-arg form
    JUnit Jupiter: message is last arg
    """
    # Matches optional "Assert." prefix then the method name then "("
    pattern = re.compile(
        r'(?:Assert\.)?'
        r'(assertTrue|assertFalse|assertNotNull|assertNull|'
        r'assertEquals|assertArrayEquals|assertSame|assertNotSame)'
        r'\s*\('
    )

    result = []
    i = 0
    while i < len(content):
        m = pattern.match(content, i)
        if m and m.start() == i:
            method = m.group(1)
            paren_open = content.index('(', i + len(m.group(0)) - 1)
            paren_close = find_matching_paren(content, paren_open)
            if paren_close == -1:
                result.append(content[i])
                i += 1
                continue
            args_str = content[paren_open + 1:paren_close]
            args = split_args(args_str)

            if method in ('assertTrue', 'assertFalse', 'assertNotNull', 'assertNull', 'assertSame', 'assertNotSame'):
                if len(args) == 2:
                    # (msg, cond) → (cond, msg)
                    new_call = f'{method}({args[1].strip()}, {args[0].strip()})'
                    result.append(new_call)
                    i = paren_close + 1
                    continue
            elif method in ('assertEquals', 'assertArrayEquals'):
                if len(args) == 3:
                    # (msg, exp, act) → (exp, act, msg)
                    new_call = f'{method}({args[1].strip()}, {args[2].strip()}, {args[0].strip()})'
                    result.append(new_call)
                    i = paren_close + 1
                    continue
            # No transformation needed for this call
            result.append(content[i])
            i += 1
        else:
            result.append(content[i])
            i += 1

    return ''.join(result)


def transform_assume_calls(content):
    """
    Assume.assumeTrue(msg, cond) → Assumptions.assumeTrue(cond, msg)
    Assume.assumeTrue(cond)      → Assumptions.assumeTrue(cond)
    """
    pattern = re.compile(r'Assume\.(assumeTrue|assumeFalse)\s*\(')
    result = []
    i = 0
    while i < len(content):
        m = pattern.match(content, i)
        if m and m.start() == i:
            method = m.group(1)
            paren_open = content.index('(', i)
            paren_close = find_matching_paren(content, paren_open)
            if paren_close == -1:
                result.append(content[i])
                i += 1
                continue
            args_str = content[paren_open + 1:paren_close]
            args = split_args(args_str)
            if len(args) == 2:
                new_call = f'Assumptions.{method}({args[1].strip()}, {args[0].strip()})'
            elif len(args) == 1:
                new_call = f'Assumptions.{method}({args[0].strip()})'
            else:
                new_call = f'Assumptions.{method}({args_str})'
            result.append(new_call)
            i = paren_close + 1
        else:
            result.append(content[i])
            i += 1
    return ''.join(result)


def transform_test_expected(content):
    """
    Convert @Test(expected = X.class) methods to @Test + assertThrows.
    Handles single-statement method bodies.
    """
    # Pattern: @Test(expected = X.class) [whitespace] public void methodName(...) [throws ...] {
    #   singleStatement;
    # }
    pattern = re.compile(
        r'@Test\s*\(\s*expected\s*=\s*([\w.]+\.class)\s*\)\s*\n'
        r'(\t| {4})(public\s+void\s+\w+\s*\(\s*\)(?:\s+throws\s+[\w,\s<>[\]]+?)?)\s*\{'
        r'\s*\n'
        r'(\t| {4,})(.*?;)\s*\n'
        r'(\t| {4})\}'
    )

    def replacer(m):
        exc_class = m.group(1)
        indent = m.group(2)
        method_header = m.group(3)
        inner_indent = m.group(4)
        statement = m.group(5).strip()
        # Remove trailing semicolon from statement for lambda
        if statement.endswith(';'):
            stmt_no_semi = statement[:-1]
        else:
            stmt_no_semi = statement
        return (f'@Test\n'
                f'{indent}{method_header} {{\n'
                f'{inner_indent}assertThrows({exc_class}, () -> {stmt_no_semi});\n'
                f'{indent}}}')

    return pattern.sub(replacer, content)


def remove_assert_qualifier(content):
    """Remove 'Assert.' prefix from assert/fail calls (we use static imports)."""
    content = re.sub(r'\bAssert\.(assert\w+|fail)\b', r'\1', content)
    content = re.sub(r'\borg\.junit\.Assert\.(assert\w+|fail)\b', r'\1', content)
    return content


def fix_duplicate_imports(content):
    """Remove duplicate import lines."""
    lines = content.split('\n')
    seen = set()
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('import ') and stripped in seen:
            continue
        seen.add(stripped)
        result.append(line)
    return '\n'.join(result)


def transform_general(content):
    """Apply general JUnit 4 → JUnit Jupiter transformations."""

    # ---- Import replacements ----
    replacements = [
        ('import org.junit.Test;',         'import org.junit.jupiter.api.Test;'),
        ('import org.junit.BeforeClass;',   'import org.junit.jupiter.api.BeforeAll;'),
        ('import org.junit.AfterClass;',    'import org.junit.jupiter.api.AfterAll;'),
        ('import org.junit.Before;',        'import org.junit.jupiter.api.BeforeEach;'),
        ('import org.junit.After;',         'import org.junit.jupiter.api.AfterEach;'),
        ('import org.junit.Ignore;',        'import org.junit.jupiter.api.Disabled;'),
        ('import org.junit.Assume;',        'import org.junit.jupiter.api.Assumptions;'),
        ('import static org.junit.Assert.*;','import static org.junit.jupiter.api.Assertions.*;'),
    ]
    for old, new in replacements:
        content = content.replace(old, new)

    # Individual static imports
    for method in ['assertEquals', 'assertTrue', 'assertFalse', 'assertNull', 'assertNotNull',
                   'assertArrayEquals', 'assertSame', 'assertNotSame', 'fail', 'assertThrows']:
        content = content.replace(
            f'import static org.junit.Assert.{method};',
            f'import static org.junit.jupiter.api.Assertions.{method};'
        )

    # "import org.junit.Assert;" → wildcard static import
    if 'import org.junit.Assert;' in content:
        content = content.replace('import org.junit.Assert;',
                                   'import static org.junit.jupiter.api.Assertions.*;')

    # ---- Annotation replacements ----
    content = re.sub(r'\b@BeforeClass\b', '@BeforeAll', content)
    content = re.sub(r'\b@AfterClass\b',  '@AfterAll', content)
    content = re.sub(r'\b@Before\b',      '@BeforeEach', content)
    content = re.sub(r'\b@After\b',       '@AfterEach', content)
    content = re.sub(r'\b@Ignore\b',      '@Disabled', content)

    # ---- @Test(expected = ...) → assertThrows ----
    content = transform_test_expected(content)

    # ---- Remove Assert. qualifier ----
    content = remove_assert_qualifier(content)

    # ---- Fix argument order for assertions ----
    content = transform_assert_calls(content)

    # ---- Assume.* → Assumptions.* ----
    content = transform_assume_calls(content)

    # ---- Remove duplicate imports ----
    content = fix_duplicate_imports(content)

    return content


# ---------------------------------------------------------------------------
# Special-case transformations
# ---------------------------------------------------------------------------

def transform_content_part_test(content):
    """ContentPartTest.java: TemporaryFolder @Rule → @TempDir."""
    content = content.replace('import org.junit.rules.TemporaryFolder;',
                               'import org.junit.jupiter.api.io.TempDir;')
    content = content.replace('import org.junit.Rule;', '')
    content = content.replace(
        '@Rule\n    public TemporaryFolder tmp = new TemporaryFolder();',
        '@TempDir\n    Path tmp;'
    )
    # tmp.newFile("x").toPath() → tmp.resolve("x")
    content = re.sub(r'tmp\.newFile\((".*?")\)\.toPath\(\)', r'tmp.resolve(\1)', content)
    # Remove blank lines left by removed Rule import
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content


def transform_abstract_cli_arg_enum_test(content):
    """
    AbstractCliArgEnumTest.java: convert @Test methods to
    @ParameterizedTest @MethodSource("data") with parameters.
    """
    # Replace imports
    content = content.replace('import org.junit.Test;',
        'import org.junit.jupiter.api.Test;\n'
        'import org.junit.jupiter.params.ParameterizedTest;\n'
        'import org.junit.jupiter.params.provider.MethodSource;')

    # Remove the constructor and instance fields, and replace with parameterized methods
    # The class structure needs a significant rework.
    # We'll do a targeted replacement of the class body.

    old_constructor_block = '''    private final E value;
    private final String expectedArgValue;
    private final int expectedEnumCount;

    protected AbstractCliArgEnumTest(E value, String expectedArgValue, int expectedEnumCount) {
        this.value = value;
        this.expectedArgValue = expectedArgValue;
        this.expectedEnumCount = expectedEnumCount;
    }

    @Test
    public void testGetArgValue() {
        assertEquals(expectedArgValue, value.getArgValue());
    }

    @Test
    public void testEnumCount() {
        assertEquals(expectedEnumCount, value.getDeclaringClass().getEnumConstants().length);
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(value instanceof CliArg);
    }

    @Test
    public void testArgValueNonEmpty() {
        assertNotNull(value.getArgValue());
        assertFalse(value.getArgValue().isEmpty());
    }'''

    new_parameterized_block = '''    @ParameterizedTest(name = "{0} -> {1}")
    @MethodSource("data")
    public void testGetArgValue(E value, String expectedArgValue, int expectedEnumCount) {
        assertEquals(expectedArgValue, value.getArgValue());
    }

    @ParameterizedTest(name = "{0} -> {1}")
    @MethodSource("data")
    public void testEnumCount(E value, String expectedArgValue, int expectedEnumCount) {
        assertEquals(expectedEnumCount, value.getDeclaringClass().getEnumConstants().length);
    }

    @ParameterizedTest(name = "{0} -> {1}")
    @MethodSource("data")
    public void testImplementsCliArg(E value, String expectedArgValue, int expectedEnumCount) {
        assertTrue(value instanceof CliArg);
    }

    @ParameterizedTest(name = "{0} -> {1}")
    @MethodSource("data")
    public void testArgValueNonEmpty(E value, String expectedArgValue, int expectedEnumCount) {
        assertNotNull(value.getArgValue());
        assertFalse(value.getArgValue().isEmpty());
    }'''

    content = content.replace(old_constructor_block, new_parameterized_block)
    return content


def transform_parameterized_subclass(content):
    """
    Convert @RunWith(Parameterized.class) subclasses of AbstractCliArgEnumTest.
    Removes: @RunWith annotation, constructor, @Parameterized.Parameters annotation.
    Updates imports.
    """
    # Remove @RunWith imports
    content = re.sub(r'import org\.junit\.runner\.RunWith;\n', '', content)
    content = re.sub(r'import org\.junit\.runners\.Parameterized;\n', '', content)
    content = re.sub(r'import java\.util\.Arrays;\n', '', content)
    content = re.sub(r'import java\.util\.Collection;\n', '', content)

    # Add new imports
    content = content.replace(
        'package net.ladenthin.llama.args;',
        'package net.ladenthin.llama.args;\n\nimport java.util.Arrays;\nimport java.util.Collection;'
    )

    # Remove @RunWith(Parameterized.class) annotation line
    content = re.sub(r'\n@RunWith\(Parameterized\.class\)\n', '\n', content)

    # Remove @Parameterized.Parameters(...) annotation line
    content = re.sub(r'\n    @Parameterized\.Parameters\([^)]*\)\n', '\n', content)

    # Remove constructor (matches pattern: "    public XTest(args) {\n        super(args);\n    }\n")
    content = re.sub(
        r'\n    public \w+\([\w]+ value, String expectedArgValue, int expectedEnumCount\) \{\n'
        r'        super\(value, expectedArgValue, expectedEnumCount\);\n'
        r'    \}\n',
        '\n',
        content
    )

    # Clean up blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content


def transform_stop_reason_test(content):
    """Rewrite StopReasonTest.java to use @ParameterizedTest @EnumSource."""
    new_content = '''// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Round-trip tests for {@link StopReason}.
 *
 * <p>The parameterised suite drives one test per enum constant: it obtains the
 * constant's {@code "stop_type"} string via {@link StopReason#getStopType()} and
 * verifies that feeding it back into {@link StopReason#fromStopType(String)} returns
 * the original constant.  The data provider is {@link StopReason#values()} so the
 * suite automatically covers any future constant added to the enum.
 *
 * <p>Edge cases (null, empty string, unknown value) are tested in separate
 * {@code @Test} methods below the round-trip test.
 */
public class StopReasonTest {

    @ParameterizedTest(name = "{0}")
    @EnumSource(StopReason.class)
    public void testRoundTrip(StopReason reason) {
        assertSame(reason, StopReason.fromStopType(reason.getStopType()));
    }

    // ------------------------------------------------------------------
    // Edge cases — tested separately from the round-trip
    // ------------------------------------------------------------------

    @Test
    public void testFromStopType_nullReturnsNone() {
        assertSame(StopReason.NONE, StopReason.fromStopType(null));
    }

    @Test
    public void testFromStopType_emptyStringReturnsNone() {
        assertSame(StopReason.NONE, StopReason.fromStopType(""));
    }

    @Test
    public void testFromStopType_unknownReturnsNone() {
        assertSame(StopReason.NONE, StopReason.fromStopType("something_else"));
    }

    @Test
    public void testEnumCount() {
        assertEquals(4, StopReason.values().length);
    }
}
'''
    return new_content


def process_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        original = f.read()

    if 'import org.junit' not in original and 'import static org.junit' not in original:
        return False

    filename = os.path.basename(path)

    # Special cases first
    if filename == 'ContentPartTest.java':
        transformed = transform_general(original)
        transformed = transform_content_part_test(transformed)
    elif filename == 'AbstractCliArgEnumTest.java':
        transformed = transform_general(original)
        transformed = transform_abstract_cli_arg_enum_test(transformed)
    elif filename == 'StopReasonTest.java':
        transformed = transform_stop_reason_test(original)
    elif 'extends AbstractCliArgEnumTest' in original:
        transformed = transform_general(original)
        transformed = transform_parameterized_subclass(transformed)
    elif filename == 'ChatExample.java':
        # Just update the @Ignore → @Disabled import
        transformed = transform_general(original)
    else:
        transformed = transform_general(original)

    if transformed != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(transformed)
        return True
    return False


def main():
    root = 'src/test/java'
    changed = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in sorted(filenames):
            if fname.endswith('.java'):
                fpath = os.path.join(dirpath, fname)
                if process_file(fpath):
                    changed.append(fpath)

    print(f"Changed {len(changed)} files:")
    for f in sorted(changed):
        print(f"  {f}")


if __name__ == '__main__':
    main()
