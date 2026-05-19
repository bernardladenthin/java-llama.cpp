// SPDX-FileCopyrightText: 2026 Gauthier Roebroeck <gauthier.roebroeck@gmail.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// Vendored 1:1 from xerial/sqlite-jdbc:
//   src/main/java/org/sqlite/util/AndroidSignatureIgnore.java
// Only deviation from the upstream file:
//   1. package org.sqlite.util  ->  package net.ladenthin.llama

package net.ladenthin.llama;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Documented
@Retention(RetentionPolicy.CLASS)
@Target({ElementType.METHOD, ElementType.CONSTRUCTOR, ElementType.TYPE, ElementType.FIELD})
public @interface AndroidSignatureIgnore {
    String explanation();
}
